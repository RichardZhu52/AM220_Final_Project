import modal
import os
import sys
import random
import subprocess
from typing import List
import tempfile # For temporary file handling
from huggingface_hub import HfApi, hf_hub_download, Repository # For HF Hub interaction
from huggingface_hub.utils import HfHubHTTPError
import torch
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
    modal.Image.debian_slim(python_version="3.10") # Specify Python version
    .apt_install("git", "git-lfs")   
    .pip_install(
        "torch==2.1.0", # Pin versions for reproducibility
        "transformers==4.35.2",
        "biopython==1.81",
        "numpy==1.26.2",
        "einops",
        "huggingface-hub",
        "geomstats",
        "scikit-bio",
        "seaborn",
        "fastdtw",
        "scanpy",
        "ete3",
        "PyQt5"
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

# --- Modal Function ---
@app.function(
    gpu="L40S",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")], # Ensure you have a valid Hugging Face token
    # Optional: Add a timeout
    timeout=86000 #24 hours
)
def embed_proteins(
    list_of_filenames: str, # afa filenames list
    hf_repo_id: str, # ID of the target Hugging Face Hub dataset repository
    model_name: str = "Synthyra/ESMplusplus_large"
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model and Tokenizer ---
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
        # Count zero entries
        num_zeros = torch.sum(distance_matrix_for_nj == 0).item()
        print(f"Number of zero entries in the distance matrix: {num_zeros}")

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
        fmt = f"{{:.{digits}f}}"
        for lbl, row in zip(labels, dm_np):
            safe_lbl = lbl.replace(" ", "_")
            row_str = " ".join(fmt.format(x) for x in row)
            phylip_file.write(f"{safe_lbl} {row_str}\n")
        for lbl, row in zip(labels, dm_np):
            # labels must be ≤64 chars, no spaces or '(),:'
            safe_lbl = lbl.replace(" ", "_")
            phylip_file.write(f"{safe_lbl} {' '.join(map(str, row))}\n")
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
            "-f", str(digits),]

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
def gpu_container_compute_poincare_maps(tree_embeddings, labels, protein_accession, k_neighbors=3, sigma = 0.10, gamma = 2.0, adaptive_sigma=False):
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

def cpu_container_compute_poincare_maps(tree_embeddings, labels, protein_accession, k_neighbors=3, sigma = 0.10, gamma = 2.0, adaptive_sigma=False):
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

# def get_treeRep_distance_matrix(distance_matrix):
#     # Assume 'your_distance_matrix' is your N x N NumPy array
#     n = distance_matrix.shape[0]
#     tree_builder = TreeRep(d=distance_matrix)
#     tree_builder.learn_tree()
#     all_pairs_sp_matrix = nx.floyd_warshall_numpy(tree_builder.G, weight='weight')
#     output_tree_distance_matrix = all_pairs_sp_matrix[0:n, 0:n]
#     return output_tree_distance_matrix

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

# Modal Entrypoint
@app.local_entrypoint()
def main(on_modal=True, poincare_hyperparameter_experiment=False, adaptive_sigma=True):
    """
    Local entrypoint to run the protein embedding function remotely via Modal.

    Args:
        fasta_file (str): Filename of the input FASTA file (relative to project root).
        model (str): Hugging Face model name.
        n_seqs (int): Number of sequences to process.
        output (str): Filename for the output embeddings (relative to project root).
    """
    #Unchanging global variables
    local_dir = "AM220_HF_Local"
    embedding_esmc_model = "Synthyra/ESMplusplus_large"
    hf_repo_id = "RichardZhu52/AM220_Final_Project"
    repo_type = "dataset"
    
    #Global variables to select script to run
    steps_to_do = ['generate_trees'] #['generate_poincare_dists','generate_trees', 'generate_embeddings']
    generation_method = "joint_CCC" #"poincare", "joint_CCC" (joint embedding optimized by cophenetic correlation coefficient), or "euclidean"

    #Output folders for poincare
    poincare_distances_folder = "Poincare_Distances_SequenceOnly_kNeighbors3_adaptiveSigma" # Folder for Poincare distances
    
    #Output folders for trees and tree construction method
    tree_construction_method = fastme #or neighbor_joining
    tree_folder = "Trees_JointCCC_SequenceOnly_kNeighbors3_adaptiveSigma_Fastme" #f"Trees_Nonhyperbolic_SequenceOnly_Fastme" # Folder for tree files, if needed
    
    #scaling factor for distance matrix
    scaling_factor = 1 # Coefficient to scale distances by

    #raw sequence alignment data
    directory = "45_combined_ali"
    all_files = os.listdir(os.path.join(LOCAL_PROJECT_PATH, directory))
    afa_files = [f for f in all_files if f.endswith(".afa")]

    # STEP 1: MAKE THE EMBEDDINGS
    if 'generate_embeddings' in steps_to_do:
        try:
            # Get existing .pt filenames from Hugging Face Hub to avoid re-processing
            existing_pt_basenames = set()
            api = HfApi()
            repo_files = api.list_repo_files(repo_id=hf_repo_id, repo_type=repo_type)
            for filename in repo_files:
                if filename.endswith(".pt"):
                    # Extract base name without extension
                    base_name, _ = os.path.splitext(filename)
                    existing_pt_basenames.add(base_name)

            original_afa_count = len(afa_files)
            afa_files_to_process = []
            for afa_filename in afa_files:
                # Extract base name from the afa filename (e.g., "AC_PF00012_1" from "AC_PF00012_1.afa")
                base_name, _ = os.path.splitext(afa_filename)
                if base_name not in existing_pt_basenames:
                    afa_files_to_process.append(afa_filename)

            print(f"Filtered list: {len(afa_files_to_process)} out of {original_afa_count} .afa files need processing.")

            list_of_filenames = [os.path.join(directory, f) for f in afa_files_to_process]
            print(f"Found {len(list_of_filenames)} .afa files to process.")

        except Exception as e:
            print(f"An error occurred while listing files: {e}")
            return

        # Call the Modal function remotely with the list of filenames
        embed_proteins.remote(
            list_of_filenames=list_of_filenames,
            hf_repo_id="RichardZhu52/AM220_Final_Project", # Replace with your HF Hub repo ID
            model_name=embedding_esmc_model
        )

    #Step 1.5: PCA Preprocessing experiments
    if 'pca_experiment' in steps_to_do:
        # Randomly sample 200 files for the experiment
        sample_size = min(200, len(afa_files)) # Ensure we don't sample more than available
        sampled_afa_files = random.sample(afa_files, sample_size)

        # Plot
        plt.figure(figsize=(12, 8))
        max_components = 0
        for i, afa_filename in enumerate(sampled_afa_files):
            # Extract the base name (e.g., "AC_PF00012_1") from the afa filename
            protein_accession, _ = os.path.splitext(os.path.basename(afa_filename))
            tree_embedding_name = f"{protein_accession}.pt"
            tree_embedding_full_path = os.path.join(local_dir, tree_embedding_name)

            # sanity check
            if not os.path.isfile(tree_embedding_full_path):
                raise FileNotFoundError(f"{tree_embedding_full_path!r} does not exist.")

            # load it
            tree_embeddings = torch.load(tree_embedding_full_path, map_location="cpu")
            components, cum_evr, evr = plot_pca_variance(tree_embeddings, protein_accession)

            # Bar plot for individual explained variance
            plt.bar(components, evr, alpha=0.6, label=protein_accession)

            # # Line plot for cumulative explained variance
            # plt.step(components, cum_evr, where='mid', color='red', label=protein_accession)

            max_components = max(max_components, len(components))

        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        if max_components <= 30:
            xticks = np.arange(1, max_components + 1)
        else:
            # every 5th component
            xticks = np.arange(1, max_components + 1, 5)
        plt.xticks(xticks)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.show()
        plt.savefig("PCA_experiment_bar_plot.png", dpi=300, bbox_inches='tight')


    #STEP 2: MAKE THE TREE
    #Get the uploaded embeddings file from the HF Hub
    if 'generate_trees' in steps_to_do or 'generate_poincare_dists' in steps_to_do:
        # clone (or init) your dataset repo locally
        embedding_repo = Repository(
            local_dir=local_dir,
            clone_from=hf_repo_id,
            repo_type=repo_type,
            git_user="RichardZhu52",
            git_email="rzhu@college.harvard.edu",
        )
        embedding_repo.git_pull()
        total_afa_files = len(afa_files)
        poincare_dist_filepath = os.path.join(LOCAL_PROJECT_PATH, poincare_distances_folder)

        if 'generate_poincare_dists' in steps_to_do:
            if generation_method == "poincare":
                #First, filter out all afa_files for which poincare distances in the poincare_distances_folder already exist
                os.makedirs(poincare_dist_filepath, exist_ok=True)
                existing_poincare_dist_files = set(os.listdir(poincare_dist_filepath))
                afa_files_for_poincare_dist_generation = [f for f in afa_files if f"{os.path.splitext(f)[0]}.pt" not in existing_poincare_dist_files]
                print(f"Filtered list: {len(afa_files_for_poincare_dist_generation)} out of {total_afa_files} .afa files need poincare dist generation.")

                # EXPERIMENT: Select 5 random proteins to analyze hyperparameters and poincare maps visually for
                if poincare_hyperparameter_experiment:
                    num_files_to_select = min(5, len(afa_files_for_poincare_dist_generation))
                    afa_files_for_poincare_dist_generation = random.sample(afa_files_for_poincare_dist_generation, num_files_to_select)

                for i, afa_filename in enumerate(afa_files_for_poincare_dist_generation):
                    # Extract the base name (e.g., "AC_PF00012_1") from the afa filename
                    protein_accession, _ = os.path.splitext(os.path.basename(afa_filename))
                    tree_embedding_name = f"{protein_accession}.pt"
                    tree_embedding_full_path = os.path.join(local_dir, tree_embedding_name)

                    # sanity check
                    if not os.path.isfile(tree_embedding_full_path):
                        raise FileNotFoundError(f"{tree_embedding_full_path!r} does not exist.")

                    # load it
                    tree_embeddings = torch.load(tree_embedding_full_path, map_location="cpu")
                    #Preprocess thorugh dimensionality reductoni
                    tree_embeddings = dimension_reduction(tree_embeddings, n_components=45)
                    # print(f"Tree dimensions: {tree_embeddings.shape}")

                    labels_hyperbolic_embedding = np.arange(len(tree_embeddings))  #Dummy labels for embeddings tensor

                    if poincare_hyperparameter_experiment:
                        for adaptive_sigma in [True, False]:
                            if on_modal:
                                embeddings_2d = gpu_container_compute_poincare_maps.remote(
                                    tree_embeddings=tree_embeddings,
                                    labels=labels_hyperbolic_embedding,
                                    protein_accession=protein_accession,
                                    adaptive_sigma=adaptive_sigma,
                                )
                            else:
                                embeddings_2d = cpu_container_compute_poincare_maps(
                                    tree_embeddings=tree_embeddings,
                                    labels=labels_hyperbolic_embedding,
                                    protein_accession=protein_accession,
                                    adaptive_sigma=adaptive_sigma
                                )
                            embeddings_2d_tensor = torch.from_numpy(embeddings_2d).double()
                            poincare_tensor_save_path = os.path.join(
                                poincare_dist_filepath, f"{protein_accession}_k_neighbors_adaptive_sig_{adaptive_sigma}.pt"
                            )
                            torch.save(embeddings_2d_tensor, poincare_tensor_save_path)

                            # 2) Visualize on Poincaré disk and save figure
                            poincare_plot_save_path = os.path.join(
                                poincare_dist_filepath, f"{protein_accession}_k_neighbors_adaptive_sig_{adaptive_sigma}.png"
                            )
                            plot_poincare_disc(x = embeddings_2d, labels = None, file_name=poincare_plot_save_path, ms=60)
                    else:
                        if on_modal:
                            embeddings_2d = gpu_container_compute_poincare_maps.remote(
                                tree_embeddings=tree_embeddings,
                                labels=labels_hyperbolic_embedding,
                                protein_accession=protein_accession,
                                adaptive_sigma=adaptive_sigma
                            )
                        else:
                            embeddings_2d = cpu_container_compute_poincare_maps(
                                tree_embeddings=tree_embeddings,
                                labels=labels_hyperbolic_embedding,
                                protein_accession=protein_accession,
                                adaptive_sigma=adaptive_sigma
                            )
                        embeddings_2d_tensor = torch.from_numpy(embeddings_2d).double()
                        poincare_tensor_save_path = os.path.join(
                            poincare_dist_filepath, f"{protein_accession}.pt"
                        )
                        torch.save(embeddings_2d_tensor, poincare_tensor_save_path)

                        # 2) Visualize on Poincaré disk and save figure
                        poincare_plot_save_path = os.path.join(
                            poincare_dist_filepath, f"{protein_accession}.png"
                        )
                        plot_poincare_disc(x = embeddings_2d, labels = None, file_name=poincare_plot_save_path, ms=100)

                    if i % 100 == 0:
                        print(f"[{i}/{total_afa_files}] Processed {afa_filename}:")

        if 'generate_trees' in steps_to_do:
            #First, filter out all afa_files for which trees in the tree_folder already exist
            tree_filepath = os.path.join(LOCAL_PROJECT_PATH, tree_folder)
            os.makedirs(tree_filepath, exist_ok=True)
            existing_tree_files = set(os.listdir(tree_filepath))
            afa_files_for_tree_generation = afa_files #[f for f in afa_files if f"{os.path.splitext(f)[0]}.tre" not in existing_tree_files]
            print(f"Filtered list: {len(afa_files_for_tree_generation)} out of {total_afa_files} .afa files need tree generation.")

            for i, afa_filename in enumerate(afa_files_for_tree_generation):
                # Extract the base name (e.g., "AC_PF00012_1") from the afa filename
                protein_accession, _ = os.path.splitext(os.path.basename(afa_filename))
                tree_embedding_name = f"{protein_accession}.pt"
                tree_embedding_full_path = os.path.join(local_dir, tree_embedding_name)

                # sanity check
                if not os.path.isfile(tree_embedding_full_path):
                    raise FileNotFoundError(f"{tree_embedding_full_path!r} does not exist.")
                
                #define phylogenetic labels
                
                # Extract Phylogenetic Labels from corresponding FASTA file 
                base_name, _ = os.path.splitext(tree_embedding_name)
                afa_filename = f"{base_name}.afa"
                afa_subdir = "45_combined_ali"
                afa_filepath = os.path.join(LOCAL_PROJECT_PATH, afa_subdir, afa_filename)
                phylo_labels = []
                with open(afa_filepath, "r") as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        phylo_labels.append(record.id) # Get the header/ID of each sequence
                print(f"Extracted {len(phylo_labels)} labels.")

                # load it
                tree_embeddings = torch.load(tree_embedding_full_path, map_location="cpu")
                if generation_method == "poincare":
                    # Load the poincare distances embeddings tensor
                    embeddings_2d_tensor = torch.load(
                        os.path.join(poincare_dist_filepath, f"{protein_accession}.pt"),
                        map_location="cpu"
                    )

                    distance_matrix = poincare_distance(embeddings_2d_tensor)
                    if i%100==0:
                        print(f"Tree {i} out of {total_afa_files} processed: {afa_filename}.")
                elif generation_method == "joint_CCC":
                    # Load the poincare distances embeddings tensor
                    poincare_embeddings = torch.load(
                        os.path.join(poincare_dist_filepath, f"{protein_accession}.pt"),
                        map_location="cpu"
                    )
                    poincare_distance_matrix = poincare_distance(poincare_embeddings)
                    euclidean_distance_matrix = calculate_similarity_matrix(
                        embeddings=tree_embeddings,
                        pt_filename=tree_embedding_name,
                        distance_metric="euclidean",
                        scaling_factor=scaling_factor, #coefficient to scale distances by
                    )

                    best_alpha = grid_search_alpha(
                        D_euc=euclidean_distance_matrix,
                        D_hyp=poincare_distance_matrix,
                        labels=phylo_labels,
                        alphas=np.linspace(0, 1, 11),
                        tree_construction_fn=tree_construction_method
                    )

                    distance_matrix = best_alpha * euclidean_distance_matrix + (1 - best_alpha) * poincare_distance_matrix

                    if i%100==0:
                        print(f"Tree {i} out of {len(afa_files_for_tree_generation)} processed: {afa_filename}.")

                    continue

                elif generation_method == "euclidean":
                    # Calculate the distance matrix using cosine similarity
                    distance_matrix = calculate_similarity_matrix(
                        embeddings=tree_embeddings,
                        pt_filename=tree_embedding_name,
                        distance_metric="cosine",
                        scaling_factor=scaling_factor, #coefficient to scale distances by
                    )

                try:
                    if distance_matrix is not None and phylo_labels and distance_matrix.shape[0] == len(phylo_labels):
                        newick_tree = tree_construction_method(distance_matrix, phylo_labels)
                        tree_filename = f"{base_name}.tre"
                        tree_filepath = os.path.join(LOCAL_PROJECT_PATH, tree_folder, tree_filename)
                        with open(tree_filepath, "w") as f:
                            f.write(newick_tree)

                except Exception as e:
                    print(f"An error occurred while reading labels or performing NJ: {e}")

    # # STEP 3: VISUALIZE THE TREE
    # newick_str, num_leaves = read_tree_size('/Users/richardzhu/AM220_Final_Project/ST_PF18072_1_IQTREE45.tre')
    # print(f"Newick: {newick_str}")
    # print(f"Number of leaves: {num_leaves}")

if __name__ == "__main__":
    # Run the main function with default parameters
    main(on_modal=False, poincare_hyperparameter_experiment=False, adaptive_sigma=True)