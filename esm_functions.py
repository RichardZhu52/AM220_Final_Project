import modal
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import random
import subprocess
from typing import List
import tempfile # For temporary file handling
from huggingface_hub import HfApi, hf_hub_download, Repository, login # For HF Hub interaction
from huggingface_hub.utils import HfHubHTTPError
import torch
from torch.cuda.amp import autocast
import time
import numpy as np
from skbio import TreeNode
from skbio import DistanceMatrix
from skbio.tree import nj
from ete3 import Tree
import torch.nn.functional as F
from Bio import SeqIO
from io import StringIO
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import geomstats.backend as gs
from geomstats.geometry.poincare_ball import PoincareBall
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import shutil
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, SamplingConfig

root = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(root, 'PoincareMaps'))
from PoincareMaps.main import compute_poincare_maps
from PoincareMaps.poincare_maps import poincare_distance
from PoincareMaps.visualize import plot_poincare_disc

sys.path.insert(0, os.path.join(root, 'TreeRep'))
from TreeRep.python_version.TreeRep import TreeRep # Assuming TreeRep.py is in the same directory or PYTHONPATH

# --- Configuration ---
# Define paths relative to the script location for mounting
LOCAL_PROJECT_PATH = os.path.dirname(__file__) if "__file__" in globals() else "."
REMOTE_PROJECT_PATH = "/project" # Using a specific remote path is often clearer

# --- Modal App Setup ---
app = modal.App(name="protein-embedder") # Give your app a name

# # Set up logging to avoid cluttering the output
# logging.set_verbosity_error()

# Mount the project directory inside the container
# Files in LOCAL_PROJECT_PATH will appear under REMOTE_PROJECT_PATH inside the container
# Define the container image with necessary libraries
image = (
    modal.Image.debian_slim(python_version="3.12") # Specify Python version
    .apt_install("git", "git-lfs")   
    .pip_install(
        "torch",#"torch==2.1.0", # Pin versions for reproducibility
        "transformers", #"transformers==4.35.2",
        "biopython", #"biopython==1.81",
        "numpy", #"numpy==1.26.2",
        "einops",
        "huggingface-hub",
        "geomstats",
        "scikit-bio",
        "seaborn",
        "fastdtw",
        "scanpy",
        "ete3",
        "PyQt5",
        "esm",
        "biotite"#"biotite==0.41.2"
        # Add accelerate for potential speedups, though not strictly required by this code
        # "accelerate"
    )
    .run_commands([
        # Install Git LFS and initialize it
        # Create the embeddings output folder ahead of time
        "git lfs install --system",
        "mkdir -p /project/PoincareEmbeddings"
    ])
    .add_local_dir(local_path=LOCAL_PROJECT_PATH, remote_path=REMOTE_PROJECT_PATH)
)

# # --- Modal Function ---
# @app.function(
#     gpu="A100-40GB",
#     image=image,
#     secrets=[modal.Secret.from_name("huggingface-secret")], # Ensure you have a valid Hugging Face token
#     # Optional: Add a timeout
#     timeout=86000 #24 hours
# )
# def new_embed_proteins(
#     list_of_filenames: str, # afa filenames list
#     hf_repo_id: str, # ID of the target Hugging Face Hub dataset repository
#     model_name: str = "Synthyra/ESMplusplus_large",
#     embedding_directory: str = ""
# ):
#     model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
#     model.eval()
#     tokenizer = None
#     protein = ESMProtein(sequence="AAAAAAAAA")
#     torch.cuda.empty_cache()
#     protein = model.generate(protein, GenerationConfig(track="structure", num_steps=4))
#     protein_tensor = model.encode(protein)
#     return protein_tensor


# --- Modal Function ---
@app.function(
    gpu="A100-40GB",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")], # Ensure you have a valid Hugging Face token
    # Optional: Add a timeout
    timeout=86000 #24 hours
)
def embed_proteins(
    list_of_filenames: str, # afa filenames list
    hf_repo_id: str, # ID of the target Hugging Face Hub dataset repository
    model_name: str = "Synthyra/ESMplusplus_large",
    embedding_directory: str = ""
):
    """
    Reads a FASTA file inside the container, removes alignment gaps,
    and embeds the first N sequences using the specified Hugging Face ESM model.
    Saves embeddings as a .pt file within the mounted project directory.

    Args:
        afa_filename (str): Filename of the input .afa FASTA file, relative to the project root.
        model_name (str): Hugging Face model identifier for ESM embeddings.

    Returns:
        str: Path to the saved embeddings file inside the container.
    """
    # AutoTokenizer might still be needed implicitly, keep import for safety
    from transformers import AutoTokenizer, AutoModel
    import time
    torch.set_default_dtype(torch.float64) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model and Tokenizer ---
    if model_name == "esm3_sm_open_v1":
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
        model.eval()
        tokenizer = None
    else:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(device)
        model.eval() # Set model to evaluation mode
        tokenizer = model.tokenizer # Get the tokenizer instance attached to the loaded model
        if tokenizer is None:
            raise RuntimeError(f"Could not retrieve tokenizer from loaded model: {model_name}")
    
    # clone (or init) your dataset repo locally
    repo = Repository(
        local_dir="AM220_HF_Local",
        clone_from=hf_repo_id,
        repo_type="dataset",
        git_user="RichardZhu52",
        git_email="rzhu@college.harvard.edu",
    )
    repo.git_pull()

    batch_size = 100

    total_trees = len(list_of_filenames)
    for i, afa_filename in enumerate(list_of_filenames):
        # Construct full paths inside the container using the remote project path
        input_path = os.path.join(REMOTE_PROJECT_PATH, afa_filename)

        # Read and clean sequences
        sequences = []
        try:
            with open(input_path, "r") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    # Remove alignment gaps
                    seq = str(record.seq).replace("-", "").replace(".", "") # Also remove '.' gaps if present
                    if seq: # Ensure sequence is not empty after removing gaps
                        sequences.append(seq)
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_path}")
            print("Files available in project root:", os.listdir(REMOTE_PROJECT_PATH))
            raise # Re-raise the exception

        if not sequences:
            raise ValueError(f"No valid sequences found or processed in {input_path}")
        
        if model_name == "esm3_sm_open_v1":
            all_embeddings = []
            def embed_one_sequence(seq, model):
                protein = ESMProtein(sequence=seq)
                protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
                protein_tensor = model.encode(protein)
                protein_tensor = protein_tensor.cpu().squeeze(0)  # (L, D)
                protein_tensor.mean(dim=0)
                return protein_tensor
            for seq in sequences:
                emb = embed_one_sequence(seq, model)
                all_embeddings.append(emb)
            # Stack into one tensor of shape (N, D)
            embeddings = torch.stack(all_embeddings, dim=0)
            print(f"ESM3 Embeddings shape: {embeddings.shape}")
        else:
            #Tokenzier then embed
            encodings = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = model(**encodings)
                # Extract embeddings (e.g., CLS token of last hidden state)
                # Note: ESM models often benefit from mean pooling over sequence length,
                # but we stick to CLS as per the original code.
                # embeddings = outputs.last_hidden_state.mean(dim=1).cpu() # Alternative: Mean pooling
                embeddings = outputs.last_hidden_state[:, 0, :].cpu() # Original: CLS token

        if i%5==0:
            print(f"Tree {i} out of {total_trees}: {afa_filename}. (Check) First element: {embeddings[0, 0].item()} Last element: {embeddings[0, -1].item()}")

        # Derive the upload filename from the input afa filename, keeping the .pt extension
        base_name = os.path.basename(afa_filename)
        file_root, _ = os.path.splitext(base_name)
        hf_upload_filename = f"{file_root}.pt" # e.g., "AC_PF00012_1.pt"
        if embedding_directory:
            # Ensure the subdirectory exists
            target_dir = os.path.join("AM220_HF_Local", embedding_directory)
            os.makedirs(target_dir, exist_ok=True)
            local_hf_dest = os.path.join(target_dir, hf_upload_filename)
        else:
            local_hf_dest = os.path.join("AM220_HF_Local", hf_upload_filename)
        torch.save(embeddings, local_hf_dest)

        # every 25 files or at end, commit & push
        if (i + 1) % batch_size == 0 or i == len(list_of_filenames) - 1:
            repo.git_add(pattern="*.pt")

            if repo.is_repo_clean():
                continue

            repo.git_commit(f"Add {min(batch_size, i+1)} embeddings")
            repo.git_push()

# def retrieve_embeddings(
#     hf_repo_id: str,
#     pt_filename: str,
#     repo_type: str = "dataset"
# ) -> torch.Tensor:
#     try: 
#         # Download the .pt file from Hugging Face Hub
#         print(f"Downloading {pt_filename} from {hf_repo_id} ({repo_type})...")
#         local_pt_path = hf_hub_download(
#             repo_id=hf_repo_id,
#             filename=pt_filename,
#             repo_type=repo_type,
#         )

#         # Load the tensor from the downloaded file
#         # Load onto CPU first to avoid potential GPU memory issues if run locally
#         # and to ensure compatibility regardless of where this function runs.
#         print("Loading embeddings tensor...")
#         embeddings = torch.load(local_pt_path, map_location='cpu')

#         if not isinstance(embeddings, torch.Tensor):
#             raise TypeError(f"Loaded file {local_pt_path} does not contain a PyTorch tensor.")
#         if embeddings.ndim != 2:
#             raise ValueError(f"Expected a 2D tensor (N x h), but got shape {embeddings.shape}")

#         print(f"Loaded tensor with shape: {embeddings.shape}")
#         N, h = embeddings.shape
#         if N == 0:
#             print("Warning: Loaded tensor is empty (0 embeddings). Returning empty tensor.")
#             return torch.empty((0, 0), dtype=embeddings.dtype)
#         else:
#             return embeddings

#     except Exception as e:
#         print(f"Error processing file {pt_filename} from repo {hf_repo_id}: {e}")
#         # Re-raise the exception to signal failure to the caller
#         raise

def calculate_similarity_matrix(
    embeddings: torch.Tensor,
    pt_filename: str,
    distance_metric: str = "cosine",
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    """
    Downloads a .pt file containing embeddings from Hugging Face Hub,
    loads it into a tensor, and calculates the pairwise cosine similarity matrix.

    Args:
        hf_repo_id (str): The Hugging Face Hub repository ID (e.g., "username/repo-name").
        pt_filename (str): The name of the .pt file within the repository (e.g., "embeddings.pt").
        repo_type (str): The type of the repository ('dataset', 'model', etc.). Defaults to 'dataset'.

    Returns:
        torch.Tensor: An N x N tensor containing pairwise cosine similarities,
                        where N is the number of embeddings in the .pt file.

    Raises:
        FileNotFoundError: If the file cannot be downloaded from the Hub.
        TypeError: If the loaded file does not contain a PyTorch tensor.
        ValueError: If the loaded tensor is not 2-dimensional (N x h).
        Exception: For other potential errors during download or processing.
    """
    try: 


        # Calculate pairwise cosine similarity
        # 1. Normalize each embedding vector to unit length.
        # 2. Compute the matrix product of the normalized tensor with its transpose.
        if distance_metric == "cosine":
            embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
            # similarity = X_norm @ X_norm^T
            similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.t())

            # Clamp values to [-1, 1] to handle potential floating point inaccuracies
            similarity_matrix = torch.clamp(similarity_matrix, -1.0, 1.0)
            similarity_matrix.fill_diagonal_(1.0) # Ensure diagonal is exactly 1.0
        elif distance_metric == "euclidean":
            # Compute pairwise Euclidean distance
            # Using broadcasting to compute the squared differences
            diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
            similarity_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))
            # Ensure diagonal is exactly zero (distance of a point to itself)
            similarity_matrix.fill_diagonal_(0.0)

        # Make sure you have a valid DISTANCE matrix here
        # If using cosine similarity, convert it:
        if distance_metric == "cosine":
            distance_matrix_for_nj = 1.0 - similarity_matrix
        elif distance_metric == "euclidean":
            distance_matrix_for_nj = similarity_matrix
        #apply scaling factor
        distance_matrix_for_nj = distance_matrix_for_nj * scaling_factor
        distance_matrix_for_nj = distance_matrix_for_nj.double()

        print(f"Calculated similarity matrix with shape: {distance_matrix_for_nj.shape}")
        return distance_matrix_for_nj

    except Exception as e:
        print(f"Error processing file {pt_filename}")
        # Re-raise the exception to signal failure to the caller
        raise

def neighbor_joining(distance_matrix, labels, min_dist = 0) -> str:
    """
    Constructs a neighbor-joining tree from a distance matrix.

    Args:
        distance_matrix (torch.Tensor): A square matrix of pairwise distances.

    Returns:
        str: Newick format string representing the neighbor-joining tree.
    """
    # Convert the PyTorch tensor to a NumPy array
    distance_matrix_np = distance_matrix.numpy()

    # Create a skbio DistanceMatrix object
    dm = DistanceMatrix(distance_matrix_np, ids=labels)

    # Construct the neighbor-joining tree
    tree = nj(dm)

    # enforce minimum branch lengthps
    if min_dist > 0: #if we want it to be impossible to have a branch length of 0
        for node in tree.postorder():
            # only adjust if a length exists and is ≤ 0
            if node.length is not None and node.length <= 0:
                node.length = min_dist

    # Use io.StringIO to capture the output of tree.write as a string
    sio = StringIO()
    tree.write(sio, format='newick') # Specify the format as 'newick'

    # Get the string value from StringIO
    newick_tree = sio.getvalue()
    sio.close() 

    return newick_tree

def fastme(distance_matrix: torch.Tensor,
           labels: List[str],
           fastme_path: str = "fastme",
           method: str = "B",
           spr: bool = True,
           digits: int = 12) -> str:
    """
    Infer a phylogenetic tree from a distance matrix using FastME.

    Args:
        distance_matrix (torch.Tensor): Square (N×N) tensor of pairwise distances.
        labels (List[str]): List of N taxon names (no spaces or '(),:').
        fastme_path (str): Path to the `fastme` executable.
        method (str): Tree-building algorithm: 'NJ', 'UNJ', 'BioNJ', 'TaxAdd_BalME', etc.
        dna_model (str): DNA substitution model (if working with DNA).
        protein_model (str): Protein model (if working with proteins).
        remove_gap (bool): If True, pass `-r` / `--remove_gap`.
        gamma_alpha (float): Gamma shape parameter (pass via `-g alpha`).
        nni (bool): If True, perform NNI moves (`-n`).
        spr (bool): If True, perform SPR moves (`-s`).
        threads (int): Number of threads (`-T`).

    Returns:
        str: Newick string of the inferred tree.
    """
    # 1. Convert tensor to NumPy and write PHYLIP distance matrix to a temp file
    if isinstance(distance_matrix, torch.Tensor):
        dm_np = distance_matrix.cpu().numpy()
    elif isinstance(distance_matrix, np.ndarray):
        dm_np = distance_matrix
    else:
        raise TypeError("distance_matrix must be a torch.Tensor or numpy.ndarray")
    n = dm_np.shape[0]
    if dm_np.shape[1] != n:
        raise ValueError("distance_matrix must be square")
    if len(labels) != n:
        raise ValueError("labels length must match matrix size")

    # Create a PHYLIP-format file
    with tempfile.NamedTemporaryFile("w+", delete=False) as phylip_file:
        name = phylip_file.name
        # First line: number of taxa
        phylip_file.write(f"{n}\n")
        # Each row: label (padded) then distances
        fmt = f"{{:.{digits}g}}"
        for lbl, row in zip(labels, dm_np):
            # sanitize label (no spaces or forbidden chars)
            safe_lbl = lbl.replace(" ", "_")[:64]
            # explicitly format every entry to `digits` decimal places
            row_str = " ".join(fmt.format(float(x)) for x in row)
            phylip_file.write(f"{safe_lbl} {row_str}\n")
    # Prepare output tree file
    out_tree = name + ".nwk"

    # 2. Build the command-line arguments
    cmd = [fastme_path,
           "-i", name,
           "-m", method]
    if spr:
        cmd += ["-s"]
    # specify where to write the tree
    cmd += ["-o", out_tree,
            "-f", str(13),]

    # 3. Run FastME
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FastME failed:\n{e.stderr.decode()}")

    # 4. Read the Newick tree and clean up
    try:
        with open(out_tree) as f:
            newick = f.read().strip()
    finally:
        # remove temp files
        os.remove(name)
        if os.path.exists(out_tree):
            os.remove(out_tree)

    return newick

@app.function(
    gpu="L40S",
    image=image,
    # Optional: Add a timeout
    # timeout=600 # 10 minutes
)
def gpu_container_compute_poincare_maps(tree_embeddings, labels, protein_accession, k_neighbors, sigma = 1.0, gamma = 2.0, adaptive_sigma=True):
    embeddings_2d, _ = compute_poincare_maps(
        features = tree_embeddings,
        labels   = labels,
        fout    = os.path.join(REMOTE_PROJECT_PATH, "PoincareEmbeddings", protein_accession),
        mode     = 'features',        # RFA built from your raw features
        k_neighbours = k_neighbors,
        distlocal    = 'minkowski',
        sigma        = sigma,
        gamma        = gamma,
        adaptive_sigma=adaptive_sigma,
        epochs       = 300,
        batchsize    = -1,            # auto-choose
        lr           = 0.1,
        burnin       = 500,
        lrm          = 1.0,
        earlystop    = 1e-4,
        cuda         = 1,             # or 1 if you have GPU
        debugplot    = False,
    )
    return embeddings_2d

def cpu_container_compute_poincare_maps(tree_embeddings, labels, protein_accession, k_neighbors, sigma = 1.0, gamma = 2.0, adaptive_sigma=False):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fout_path = os.path.join(temp_dir, protein_accession)
        embeddings_2d, _ = compute_poincare_maps(
            features = tree_embeddings,
            labels   = labels,
            fout    = os.path.join(LOCAL_PROJECT_PATH, "Trees_Playground", protein_accession), #temp_fout_path,
            mode     = 'features',        # RFA built from your raw features
            k_neighbours = k_neighbors, #5
            distlocal    = 'minkowski',
            sigma        = sigma,
            gamma        = gamma,
            adaptive_sigma=adaptive_sigma,
            epochs       = 300,
            batchsize    = -1,            # auto-choose
            lr           = 0.1,
            burnin       = 500,
            lrm          = 1.0,
            earlystop    = 1e-4,
            cuda         = 0,             # or 1 if you have GPU
            debugplot    = False,
        )
    return embeddings_2d

def read_tree_size(tre_filepath):
    """
    Reads a Newick tree, saves a visualization image, returns Newick string and leaf count.

    Args:
        tre_filepath (str): Path to the .tre file.

    Returns:
        tuple: (newick_tree (str), size (int))
    """
    from ete3.treeview import TreeStyle
    # Read the Newick string from file
    with open(tre_filepath, 'r') as f:
        newick_tree = f.read().strip()

    # Ensure it ends with a semicolon for robust parsing
    if not newick_tree.endswith(';'):
        newick_tree += ';'

    # Parse the tree with skbio to count leaves
    parsed_tree_skbio = TreeNode.read(StringIO(newick_tree), format='newick')
    size = len(list(parsed_tree_skbio.tips()))

    # Visualize and save using ete3 if available
    try:
        # Parse the tree with ete3
        ete_tree = Tree(newick_tree)

        # Define output image path (same directory, base name + .png)
        base_path, _ = os.path.splitext(tre_filepath)
        img_filepath = f"{base_path}_visualization.png"

        # Create a basic tree style
        ts = TreeStyle()
        ts.show_leaf_name = True # Display leaf names
        # Optional: Add more style customizations here if needed
        # ts.mode = "c" # Circular layout example

        # Render the tree to the image file
        print(f"Saving tree visualization to: {img_filepath}")
        # Use dpi for higher resolution if desired, e.g., dpi=300
        ete_tree.render(img_filepath, tree_style=ts)

    except Exception as e:
        # Catch potential errors during ete3 processing/rendering
        print(f"Warning: Failed to generate tree visualization using ete3: {e}")

    return newick_tree, size


def plot_pca_variance(X, protein_name, n_components=None):
    """
    Perform PCA on data and plot explained variance.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to decompose.
    n_components : int or None (default=None)
        Number of principal components to compute. If None, all components are used.
    figsize : tuple (width, height), optional
        Size of the matplotlib figure.

    """
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Explained variance ratio for each component
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    components = np.arange(1, len(evr) + 1)

    return components, cum_evr, evr

def dimension_reduction(embeddings, n_components=45):
    """
    Perform PCA on the embeddings to reduce dimensionality.

    Args:
        embeddings (torch.Tensor): The input embeddings tensor.
        n_components (int): Number of components to keep.

    Returns:
        torch.Tensor: The reduced embeddings.
    """
    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    return torch.tensor(reduced_embeddings, device=embeddings.device)

def get_treeRep_distance_matrix(distance_matrix):
    # Assume 'your_distance_matrix' is your N x N NumPy array
    n = distance_matrix.shape[0]
    tree_builder = TreeRep(d=distance_matrix)
    tree_builder.learn_tree()
    all_pairs_sp_matrix = nx.floyd_warshall_numpy(tree_builder.G, weight='weight')
    output_tree_distance_matrix = all_pairs_sp_matrix[0:n, 0:n]
    return output_tree_distance_matrix

def compute_cophenetic_matrix(tree: TreeNode, labels: list) -> np.ndarray:
    """
    Given a scikit-bio TreeNode, compute the full cophenetic distance matrix.
    """
    tip_map = {tip.name: tip for tip in tree.tips()}

    n = len(labels)
    cophe = np.zeros((n, n))

    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            node_a = tip_map[a]
            node_b = tip_map[b]
            cophe[i, j] = tree.distance(node_a, node_b)
    return cophe

def grid_search_alpha(D_euc: np.ndarray,
                      D_hyp: np.ndarray,
                      labels: list,
                      alphas: np.ndarray,
                      tree_construction_fn
                      ):
    """
    Grid-search alpha to maximize CCC (= Pearson r) between
    fused distances and tree cophenetic distances.
    
    Returns: (alphas, cccs, best_alpha)
    """
    n = len(labels)
    cccs = []

    D_euc_t = torch.as_tensor(D_euc)
    D_hyp_t = torch.as_tensor(D_hyp)

    #normalize the distance matrices to [0,1] (min distance is already 0)
    D_euc_t = D_euc_t / D_euc_t.max()
    D_hyp_t = D_hyp_t / D_hyp_t.max()

    for alpha in alphas:
        # fuse directly in torch, and cast to float32
        D_fused_t = (alpha * D_euc_t + (1 - alpha) * D_hyp_t).to(torch.float32)
        newick = tree_construction_fn(D_fused_t, labels)
        tree = TreeNode.read([newick])
        cophe = compute_cophenetic_matrix(tree, labels)
        iu = np.triu_indices(n, k=1)
        r, _ = pearsonr(D_fused_t[iu], cophe[iu])
        cccs.append(r)

    cccs = np.array(cccs)
    best_alpha = alphas[np.argmax(cccs)]
    print(f"Best alpha: {best_alpha}")
    return best_alpha

class Experiments:
    """
    Encapsulates generation of embeddings, PCA preprocessing experiments,
    Poincaré distance computations, tree construction, and tree visualization.
    """
    def __init__(
        self,
        local_dir: str = "AM220_HF_Local",
        hf_repo_id: str = "RichardZhu52/AM220_Final_Project",
        repo_type: str = "dataset",
        embedding_model: str = "Synthyra/ESMplusplus_large",
        on_modal: bool = True
    ):
        self.local_dir = local_dir
        self.hf_repo_id = hf_repo_id
        self.repo_type = repo_type
        self.embedding_model = embedding_model
        self.on_modal = on_modal
        self.api = HfApi()
        self._collect_fasta_files()

    def _collect_fasta_files(self, fasta_dir = "45_combined_ali"):
        all_files = os.listdir(fasta_dir)
        self.afa_files = [f for f in all_files if f.endswith(".afa")]
        return all_files

    def generate_fasta_test_set(self, fasta_dir, fasta_test_set_dir, test_num):
        if not os.path.exists(fasta_test_set_dir):
            os.makedirs(fasta_test_set_dir)

        num_to_select = min(test_num, len(self.afa_files))
        selected_files = random.sample(self.afa_files, num_to_select)

        for file_name in selected_files:
            source_path = os.path.join(fasta_dir, file_name) # Assuming afa_files are in self.fasta_dir
            destination_path = os.path.join(fasta_test_set_dir, file_name)
            # Copy the file
            # Using shutil.copy for simplicity, ensure it's imported if not already
            try:
                shutil.copy(source_path, destination_path)
                print(f"Copied {file_name} to {fasta_test_set_dir}")
            except Exception as e:
                print(f"Error copying {file_name}: {e}")
        
        print(f"Selected {num_to_select} files and saved them to {fasta_test_set_dir}")
        return selected_files
    
    def generate_embeddings(self, 
                            list_of_filenames: str, # afa filenames list
                            embedding_directory: str = "",
                            skip_existing: bool = True,
                            fasta_dir: str = "45_combined_ali"):
        """
        Generate embeddings for .afa files and upload to HF hub.
        """
        existing = set()
        if skip_existing:
            repo_files = self.api.list_repo_files(repo_id=self.hf_repo_id, repo_type=self.repo_type)
            
            # Determine the target directory path on the Hub.
            # An empty embedding_directory means the root.
            # os.path.normpath('') gives '.', but for Hub paths, an empty string is clearer for root.
            hub_target_dir = embedding_directory.strip("/") # Remove leading/trailing slashes for consistency

            for filepath_on_hub in repo_files:
                if not filepath_on_hub.endswith(".pt"):
                    continue

                # Normalize path, e.g., remove potential leading "./"
                normalized_filepath_on_hub = filepath_on_hub
                if normalized_filepath_on_hub.startswith("./"):
                    normalized_filepath_on_hub = normalized_filepath_on_hub[2:]

                file_dir_on_hub = os.path.dirname(normalized_filepath_on_hub)
                file_basename_on_hub = os.path.basename(normalized_filepath_on_hub)

                # Match directory:
                # If hub_target_dir is empty (root), file_dir_on_hub should also be empty.
                # If hub_target_dir is specified, file_dir_on_hub should match it.
                if file_dir_on_hub == hub_target_dir:
                    base_name, _ = os.path.splitext(file_basename_on_hub)
                    existing.add(base_name)

        to_process = [f for f in list_of_filenames
                      if os.path.splitext(f)[0] not in existing]
        print(f"Processing {len(to_process)} files for embeddings.")
        paths = [os.path.join(fasta_dir, f) for f in to_process]
        if len(to_process) == 0:
            print("No new files to process.")
            return
        # Call remote embed function
        embed_proteins.remote(
            list_of_filenames=paths,
            hf_repo_id=self.hf_repo_id,
            model_name=self.embedding_model,
            embedding_directory=embedding_directory
        )

    def pca_experiment(self, embedding_directory, sample_size: int = 200, n_components: int = 45, save_plot: bool = True):
        """
        Randomly sample sequences, compute PCA variance, and plot diagnostics.
        """
        sample = random.sample(self.afa_files, min(sample_size, len(self.afa_files)))
        plt.figure(figsize=(12, 8))
        max_comps = 0
        for afa in sample:
            base, _ = os.path.splitext(afa)
            if embedding_directory:
                emb_path = os.path.join(self.local_dir, embedding_directory, base + ".pt")
            else:
                emb_path = os.path.join(self.local_dir, base + ".pt")
            if not os.path.isfile(emb_path):
                raise FileNotFoundError(f"{emb_path} not found.")
            emb = torch.load(emb_path, map_location="cpu")
            comps, cum_evr, evr = plot_pca_variance(emb, base, n_components)
            plt.bar(comps, evr, alpha=0.6, label=base)
            max_comps = max(max_comps, len(comps))

        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        xticks = np.arange(1, max_comps + 1 if max_comps <= 30 else max_comps + 1, 
                           1 if max_comps <= 30 else 5)
        plt.xticks(xticks)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        if save_plot:
            plt.savefig("PCA_experiment.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_poincare_distances(self, list_of_filenames, poincare_folder, structure_embedding_directory, sequence_embedding_directory, poincare_method, n_components, k_neighbors, sigma, adaptive_sigma: bool = True):
        """
        Compute Poincaré embeddings and save distance tensors and plots.
        """
        os.makedirs(poincare_folder, exist_ok=True)
        to_do = []
        existing = {f for f in os.listdir(poincare_folder) if f.endswith('.pt')}
        for afa in list_of_filenames:
            base = os.path.splitext(afa)[0]
            target = f"{base}.pt"
            if target not in existing:
                to_do.append(afa)

        for i, afa in enumerate(to_do):
            base, _ = os.path.splitext(afa)
            if poincare_method == "structure" or poincare_method == "structure_sequence_concat":
                if structure_embedding_directory:
                    structure_emb_path = os.path.join(self.local_dir, structure_embedding_directory, base + ".pt")
                else:
                    structure_emb_path = os.path.join(self.local_dir, base + ".pt")
                if not os.path.isfile(structure_emb_path):
                    raise FileNotFoundError(f"{structure_emb_path} not found.")
                structure_emb = torch.load(structure_emb_path, map_location="cpu")
            if poincare_method == "sequence" or poincare_method == "structure_sequence_concat":
                if sequence_embedding_directory:
                    sequence_emb_path = os.path.join(self.local_dir, sequence_embedding_directory, base + ".pt")
                else:
                    sequence_emb_path = os.path.join(self.local_dir, base + ".pt")
                if not os.path.isfile(sequence_emb_path):
                    raise FileNotFoundError(f"{sequence_emb_path} not found.")
                sequence_emb = torch.load(sequence_emb_path, map_location="cpu")
            if poincare_method == "structure_sequence_concat":
                emb = torch.cat((structure_emb, sequence_emb), dim=1)
            elif poincare_method == "structure":
                emb = structure_emb
            elif poincare_method == "sequence":
                emb = sequence_emb
            emb_reduced = dimension_reduction(emb, n_components=n_components)
            labels = np.arange(len(emb_reduced))

            method = gpu_container_compute_poincare_maps if self.on_modal else cpu_container_compute_poincare_maps
            res = method.remote(tree_embeddings=emb_reduced, labels=labels,
                                protein_accession=base, adaptive_sigma=adaptive_sigma, k_neighbors = k_neighbors,
                                sigma = sigma)
            tensor = torch.from_numpy(res).double()
            torch.save(tensor, os.path.join(poincare_folder, f"{base}.pt"))
            plot_poincare_disc(res, None,
                                os.path.join(poincare_folder, f"{base}.png"), ms=100)

            if i % 25 == 0:
                print(f"[{i}/{len(to_do)}] Processed {afa} for Poincaré.")
    
    def generate_trees(self, list_of_filenames, tree_folder, fasta_dir, sequence_embedding_directory, structure_embedding_directory, generation_method, tree_method, poincare_folder=None, rescale_poincare=False, other_poincare_folder=None): #other_poincare_folder only used for cophenetic correlation calculations
        """
        Construct phylogenetic trees based on chosen generation_method.
        """
        # Pull latest embeddings
        repo = Repository(local_dir=self.local_dir,
                          clone_from=self.hf_repo_id,
                          repo_type=self.repo_type)
        repo.git_pull()
        os.makedirs(tree_folder, exist_ok=True)

        for i, afa in enumerate(list_of_filenames):
            base, _ = os.path.splitext(afa)
            if sequence_embedding_directory:
                emb_path = os.path.join(self.local_dir, sequence_embedding_directory, base + ".pt")
            else:
                emb_path = os.path.join(self.local_dir, base + ".pt")
            if not os.path.isfile(emb_path):
                raise FileNotFoundError(f"{emb_path} not found.")

            # Load phylogenetic labels
            seqs = list(SeqIO.parse(os.path.join(fasta_dir, afa), "fasta"))
            labels = [r.id for r in seqs]

            seq_emb = torch.load(emb_path, map_location="cpu")
            if generation_method == "cosine":
                dist = calculate_similarity_matrix(seq_emb, pt_filename=base, distance_metric="cosine")
            elif generation_method == "euclidean":
                dist = calculate_similarity_matrix(seq_emb, pt_filename=base, distance_metric="euclidean")
            elif generation_method == "euclidean_structure_sequence_CCC":
                seq_dist = calculate_similarity_matrix(seq_emb, pt_filename=base, distance_metric="euclidean")
                if structure_embedding_directory:
                    structure_emb_path = os.path.join(self.local_dir, structure_embedding_directory, base + ".pt")
                else:
                    structure_emb_path = os.path.join(self.local_dir, base + ".pt")
                if not os.path.isfile(structure_emb_path):
                    raise FileNotFoundError(f"{structure_emb_path} not found.")
                structure_emb = torch.load(structure_emb_path, map_location="cpu")
                structure_dist = calculate_similarity_matrix(structure_emb, pt_filename=base, distance_metric="euclidean")
                alpha = grid_search_alpha(structure_dist, seq_dist, labels,
                                             alphas=np.linspace(0, 1, 6),
                                             tree_construction_fn=tree_method)
                dist = alpha * structure_dist + (1 - alpha) * seq_dist
            else:
                hyp = torch.load(os.path.join(poincare_folder, f"{base}.pt"), map_location="cpu")
                D_hyp = poincare_distance(hyp, rescale=rescale_poincare)
                if generation_method == "poincare":
                    dist = D_hyp
                elif generation_method == "structure_sequence_CCC":
                    hyp_other = torch.load(os.path.join(other_poincare_folder, f"{base}.pt"), map_location="cpu")
                    D_hyp_other = poincare_distance(hyp_other, rescale=rescale_poincare)

                    alpha = grid_search_alpha(D_hyp_other, D_hyp, labels,
                                                 alphas=np.linspace(0, 1, 6),
                                                 tree_construction_fn=tree_method)
                    dist = alpha * D_hyp_other + (1 - alpha) * D_hyp
            newick = tree_method(dist, labels)
            with open(os.path.join(tree_folder, base + ".tre"), "w") as f:
                f.write(newick)

            if i % 100 == 0:
                print(f"[{i}] Tree generated for {afa}.")
    
    def visualize_tree(self, tree_path: str):
        """
        Read and print newick string and leaf count from a tree file.
        """
        newick_str, num_leaves = read_tree_size(tree_path)
        print(f"Newick: {newick_str}\nLeaves: {num_leaves}")

# Modal Entrypoint
@app.local_entrypoint()
def main(on_modal=True):
    # #Parameters to test for Euclidean experiments
    # for distance_metric in ["cosine", "euclidean"]:
    #     for construction_method in [fastme, neighbor_joining]:
    #         exp = Experiments(on_modal=on_modal) #embedding_model="esm3_sm_open_v1")

    #         # --- Configuration for this specific run ---
    #         # Source directory of all .afa files
    #         fasta_source_dir = "45_combined_ali"
    #         fasta_test_set_dir = "fasta_test_set_200_0507"
    #         num_test_files = 200

    #         # HF Hub subdirectory and local directory name for esm3_sm_open_v1 embeddings
    #         embedding_directory_name = f"ESMC_Sequence_Embeddings" # "embeddings_esm3_300_0507"
            
    #         # Local folder for generated Newick tree files
    #         tree_folder_name = f"trees_esmc_euclidean_0507_dist_{distance_metric}_construction_{construction_method}" #"trees_esm3_300_0507"
    #         # Method to generate distance matrix for tree building (from Poincare embeddings)
    #         tree_generation_method = distance_metric 
    #         # Algorithm for tree construction (e.g., fastme or neighbor_joining)
    #         tree_construction_algorithm = construction_method

    #         # --- Step 1: Sample 300 files for testing ---
    #         # sampled_filenames_basenames = exp.generate_fasta_test_set(
    #         #     fasta_dir=fasta_source_dir,
    #         #     fasta_test_set_dir=fasta_test_set_dir,
    #         #     test_num=num_test_files
    #         # )
    #         sampled_filenames_basenames = exp._collect_fasta_files(fasta_test_set_dir)

    #         # --- Step 2: Generate embeddings for these sampled files using esm3_sm_open_v1 ---
    #         print(f"Generating esm3_sm_open_v1 embeddings...")
    #         exp.generate_embeddings(
    #             list_of_filenames=sampled_filenames_basenames,
    #             embedding_directory=embedding_directory_name, # Subdirectory on HF Hub
    #             skip_existing=True, # Recommended
    #             fasta_dir=fasta_test_set_dir # Location of the .afa files to be embedded
    #         )

    #         # --- Step 3: Pull embeddings and generate Poincaré distances ---
    #         print(f"Ensuring local Hugging Face repository '{exp.local_dir}' is up-to-date...")
    #         try:
    #             repo = Repository(local_dir=exp.local_dir, clone_from=exp.hf_repo_id, repo_type=exp.repo_type)
    #             repo.git_pull()
    #             print("Local repository updated.")
    #         except Exception as e:
    #             print(f"Could not pull from Hugging Face Hub repository: {e}")

    #         # --- Step 4: Generate trees using Poincare distances ---
    #         print(f"Generating phylogenetic trees for {len(sampled_filenames_basenames)} files...")
    #         exp.generate_trees(
    #             list_of_filenames=sampled_filenames_basenames,
    #             tree_folder=tree_folder_name,
    #             sequence_embedding_directory=embedding_directory_name,
    #             fasta_dir=fasta_test_set_dir, # Source of .afa files for labels
    #             generation_method=tree_generation_method,
    #             tree_method=tree_construction_algorithm
    #         )
    #         print(f"Tree generation complete. Results in '{tree_folder_name}'.")

    # #Parameters to test for Poincare experiments
    # k_neighbors = [5,7,9]
    # pca_components = [4,10,45]
    # poincare_rescaling = ["True", "False"]
    # for k in k_neighbors:
    #     for pca in pca_components:
    #         for s in poincare_rescaling:
    #             print(f"Running with k_neighbors={k}, pca_components={pca}, rescale={s}")

    #             exp = Experiments(on_modal=on_modal) #embedding_model="esm3_sm_open_v1")

    #             # --- Configuration for this specific run ---
    #             # Source directory of all .afa files
    #             fasta_source_dir = "45_combined_ali"
    #             fasta_test_set_dir = "fasta_test_set_200_0507"
    #             num_test_files = 200

    #             # HF Hub subdirectory and local directory name for esm3_sm_open_v1 embeddings
    #             embedding_directory_name = f"ESMC_Sequence_Embeddings" # "embeddings_esm3_300_0507"
                
    #             # Local folder for Poincare distance tensors and plots
    #             poincare_folder_name = f"poincare_dists_esmc_0507_k_{k}_pca_{pca}_rescaling_{s}" 
    #             poincare_n_components_pca = pca # PCA components before Poincare embedding
    #             poincare_k_neighbors = k      # k-neighbors for Poincare RAG
    #             poincare_Sigma = 1.0
    #             poincare_adaptive_sigma = False # Use adaptive sigma for Poincare RAG
                
    #             # Local folder for generated Newick tree files
    #             tree_folder_name = f"trees_hyperbolic_esmc_0507_k_{k}_pca_{pca}_rescaling_{s}"
    #             # Method to generate distance matrix for tree building (from Poincare embeddings)
    #             tree_generation_method = "poincare" 
    #             # Algorithm for tree construction (e.g., fastme or neighbor_joining)
    #             tree_construction_algorithm = fastme
    #             if s == "True":
    #                 rescale_poincare = True
    #             else:
    #                 rescale_poincare = False

    #             # --- Step 1: Sample 300 files for testing ---
    #             # sampled_filenames_basenames = exp.generate_fasta_test_set(
    #             #     fasta_dir=fasta_source_dir,
    #             #     fasta_test_set_dir=fasta_test_set_dir,
    #             #     test_num=num_test_files
    #             # )
    #             sampled_filenames_basenames = exp._collect_fasta_files(fasta_test_set_dir)

    #             # --- Step 2: Generate embeddings for these sampled files using esm3_sm_open_v1 ---
    #             print(f"Generating esm3_sm_open_v1 embeddings...")
    #             exp.generate_embeddings(
    #                 list_of_filenames=sampled_filenames_basenames,
    #                 embedding_directory=embedding_directory_name, # Subdirectory on HF Hub
    #                 skip_existing=True, # Recommended
    #                 fasta_dir=fasta_test_set_dir # Location of the .afa files to be embedded
    #             )

    #             # --- Step 3: Pull embeddings and generate Poincaré distances ---
    #             print(f"Ensuring local Hugging Face repository '{exp.local_dir}' is up-to-date...")
    #             try:
    #                 repo = Repository(local_dir=exp.local_dir, clone_from=exp.hf_repo_id, repo_type=exp.repo_type)
    #                 repo.git_pull()
    #                 print("Local repository updated.")
    #             except Exception as e:
    #                 print(f"Could not pull from Hugging Face Hub repository: {e}")

    #             print(f"Generating Poincaré distances...")
    #             exp.generate_poincare_distances(
    #                 list_of_filenames=sampled_filenames_basenames,
    #                 poincare_folder=poincare_folder_name,
    #                 embedding_directory=embedding_directory_name,
    #                 n_components=poincare_n_components_pca,
    #                 k_neighbors=poincare_k_neighbors,
    #                 adaptive_sigma=poincare_adaptive_sigma,
    #                 sigma=poincare_Sigma # Sigma for Poincare RAG
    #             )
    #             print(f"Poincaré distance generation complete. Results in '{poincare_folder_name}'.")

    #             # --- Step 4: Generate trees using Poincare distances ---
    #             print(f"Generating phylogenetic trees for {len(sampled_filenames_basenames)} files...")
    #             exp.generate_trees(
    #                 list_of_filenames=sampled_filenames_basenames,
    #                 tree_folder=tree_folder_name,
    #                 sequence_embedding_directory=embedding_directory_name,
    #                 fasta_dir=fasta_test_set_dir, # Source of .afa files for labels
    #                 generation_method=tree_generation_method,
    #                 tree_method=tree_construction_algorithm,
    #                 poincare_folder=poincare_folder_name, # Source of Poincare distance matrices
    #                 rescale_poincare=rescale_poincare
    #             )
    #             print(f"Tree generation complete. Results in '{tree_folder_name}'.")

    #Euclidean structure embedding experiments
    for embed_method in ["structure_sequence_CCC"]:
        exp = Experiments(on_modal=on_modal)

        # --- Configuration for this specific run ---
        # Source directory of all .afa files
        fasta_test_set_dir = "fasta_test_set_200_0507"

        sequence_embedding_directory_name = f"ESMC_Sequence_Embeddings"
        structure_embedding_directory_name = f"embeddings_esm3_all_trees_0507"
        
        # Local folder for generated Newick tree files
        tree_folder_name = f"trees_structure_euclidean_0509_embed_method_{embed_method}" #"trees_esm3_300_0507"
        # Method to generate distance matrix for tree building
        if embed_method == "structure_sequence_CCC":
            tree_generation_method = "euclidean_structure_sequence_CCC"
        else:
            tree_generation_method = "euclidean" 
        # Algorithm for tree construction (e.g., fastme or neighbor_joining)
        tree_construction_algorithm = fastme

        # --- Step 1: Sample 300 files for testing ---
        sampled_filenames_basenames = exp._collect_fasta_files(fasta_test_set_dir)

        # --- Step 3: Pull embeddings and generate Poincaré distances ---
        print(f"Ensuring local Hugging Face repository '{exp.local_dir}' is up-to-date...")
        try:
            repo = Repository(local_dir=exp.local_dir, clone_from=exp.hf_repo_id, repo_type=exp.repo_type)
            repo.git_pull()
            print("Local repository updated.")
        except Exception as e:
            print(f"Could not pull from Hugging Face Hub repository: {e}")

        # --- Step 4: Generate trees using Poincare distances ---
        print(f"Generating phylogenetic trees for {len(sampled_filenames_basenames)} files...")
        exp.generate_trees(
            list_of_filenames=sampled_filenames_basenames,
            tree_folder=tree_folder_name,
            sequence_embedding_directory=sequence_embedding_directory_name,
            structure_embedding_directory=structure_embedding_directory_name,
            fasta_dir=fasta_test_set_dir, # Source of .afa files for labels
            generation_method=tree_generation_method,
            tree_method=tree_construction_algorithm
        )
        print(f"Tree generation complete. Results in '{tree_folder_name}'.")

    #Parameters to test for structure embedding experiments
    k_neighbors = [5]
    embedding_methods = ['structure','structure_sequence_CCC'] #not doing structure sequence concat because the distances are on different scales, structure distance washed out sequence distance
    for k in k_neighbors:
        for embed_method in embedding_methods:
            try:
                print(f"Running with k_neighbors={k}, embedding_method={embed_method}")

                exp = Experiments(on_modal=on_modal) #embedding_model="esm3_sm_open_v1")

                # --- Configuration for this specific run ---
                # Source directory of all .afa files
                fasta_test_set_dir = "fasta_test_set_200_0507"

                # HF Hub subdirectory and local directory name for esm3_sm_open_v1 embeddings
                sequence_embedding_directory_name = f"ESMC_Sequence_Embeddings"
                structure_embedding_directory_name = f"embeddings_esm3_all_trees_0507"
                
                # Local folder for Poincare distance tensors and plots
                poincare_folder_name = f"poincare_dists_structure_0509_k_{k}_embedding_method_{embed_method}_adaptive_sigma" 
                poincare_n_components_pca = 10 # PCA components before Poincare embedding
                poincare_k_neighbors = k      # k-neighbors for Poincare RAG
                poincare_Sigma = None
                poincare_adaptive_sigma = True # Use adaptive sigma for Poincare RAG
                
                # Local folder for generated Newick tree files
                # Method to generate distance matrix for tree building (from Poincare embeddings)
                if embed_method == "structure_sequence_CCC":
                    tree_generation_method = "structure_sequence_CCC"
                else:
                    tree_generation_method = "poincare" 
                # Algorithm for tree construction (e.g., fastme or neighbor_joining)
                tree_construction_algorithm = fastme
                rescale_poincare = True
                tree_folder_name = f"trees_structure_hyperbolic_0509_k_{k}_embedding_method_{embed_method}_adaptive_sigma"

                # --- Step 1: Sample 300 files for testing ---
                sampled_filenames_basenames = exp._collect_fasta_files(fasta_test_set_dir)

                # --- Step 2: Generate embeddings for these sampled files using esm3_sm_open_v1 ---
                print(f"Generating esm3_sm_open_v1 embeddings...")
                exp.generate_embeddings(
                    list_of_filenames=sampled_filenames_basenames,
                    embedding_directory=structure_embedding_directory_name, # Subdirectory on HF Hub
                    skip_existing=True, # Recommended
                    fasta_dir=fasta_test_set_dir # Location of the .afa files to be embedded
                )

                # --- Step 3: Pull embeddings and generate Poincaré distances ---
                print(f"Ensuring local Hugging Face repository '{exp.local_dir}' is up-to-date...")
                try:
                    repo = Repository(local_dir=exp.local_dir, clone_from=exp.hf_repo_id, repo_type=exp.repo_type)
                    repo.git_pull()
                    print("Local repository updated.")
                except Exception as e:
                    print(f"Could not pull from Hugging Face Hub repository: {e}")
                structure_distances = f"poincare_dists_structure_0509_k_{k}_embedding_method_structure_adaptive_sigma"
                if embed_method == "structure_sequence_CCC" and os.path.exists(structure_distances):
                    print(f"Using existing Poincaré distances from '{structure_distances}'...")
                else:
                    print(f"Generating Poincaré distances...")
                    exp.generate_poincare_distances(
                        list_of_filenames=sampled_filenames_basenames,
                        poincare_folder=poincare_folder_name,
                        structure_embedding_directory=structure_embedding_directory_name,
                        sequence_embedding_directory=sequence_embedding_directory_name,
                        poincare_method = embed_method,
                        n_components=poincare_n_components_pca,
                        k_neighbors=poincare_k_neighbors,
                        adaptive_sigma=poincare_adaptive_sigma,
                        sigma=poincare_Sigma # Sigma for Poincare RAG
                    )
                    print(f"Poincaré distance generation complete. Results in '{poincare_folder_name}'.")

                # --- Step 4: Generate trees using Poincare distances ---
                print(f"Generating phylogenetic trees for {len(sampled_filenames_basenames)} files...")
                if embed_method == "structure_sequence_CCC":
                    exp.generate_trees(
                        list_of_filenames=sampled_filenames_basenames,
                        tree_folder=tree_folder_name,
                        sequence_embedding_directory=None, #won't be used
                        structure_embedding_directory=None,
                        fasta_dir=fasta_test_set_dir, # Source of .afa files for labels
                        generation_method=tree_generation_method,
                        tree_method=tree_construction_algorithm,
                        poincare_folder=f"poincare_dists_structure_0509_k_{k}_embedding_method_structure_adaptive_sigma",
                        other_poincare_folder=f"poincare_dists_esmc_0507_k_5_pca_10_rescaling_True",
                        rescale_poincare=rescale_poincare
                    )
                else:
                    exp.generate_trees(
                        list_of_filenames=sampled_filenames_basenames,
                        tree_folder=tree_folder_name,
                        sequence_embedding_directory=None, #won't be used
                        structure_embedding_directory=None,
                        fasta_dir=fasta_test_set_dir, # Source of .afa files for labels
                        generation_method=tree_generation_method,
                        tree_method=tree_construction_algorithm,
                        poincare_folder=poincare_folder_name, # Source of Poincare distance matrices
                        rescale_poincare=rescale_poincare
                    )
                print(f"Tree generation complete. Results in '{tree_folder_name}'.")
            except Exception as e:
                print(f"Error during processing: {e}")
                # Save an error report file naming k_neighbors and embedding_method
                error_fname = f"error_k_neighbors_{k}_embedding_method_{embed_method}.txt"
                with open(error_fname, "w") as err_f:
                    err_f.write(f"Error encountered for k_neighbors={k}, embedding_method={embed_method}:\n")
                    err_f.write(str(e))
                continue

if __name__ == "__main__":
    # Run the main function with default parameters
    main(on_modal=False)