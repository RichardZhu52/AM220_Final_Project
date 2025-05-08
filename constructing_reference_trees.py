import glob
import os
import subprocess
import logging
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_msas_for_uniprot(
    uniprot_id: str,
    bucket: str = "openfold",
    local_root: str = "./msas/uniprot",
    region_name: str = "us-east-1"
) -> list[str]:
    """
    Download the three .a3m MSA files for a given uniprot ID from openfold S3.

    Args:
        uniprot_id (str): e.g. "1ABC"
        bucket (str): S3 bucket name (default: "openfold").
        local_root (str): Base local folder for downloads.
        region_name (str): AWS region (needed for some unsigned setups).

    Returns:
        List[str]: Absolute local paths of the downloaded .a3m files.
    """
    # 1) Create an anonymous (unsigned) S3 client
    s3 = boto3.client(
        "s3",
        region_name=region_name,
        config=Config(signature_version=UNSIGNED)
    )  # uses UNSIGNED to mimic --no-sign-request :contentReference[oaicite:0]{index=0}

    # 2) Build the S3 prefix for this uniprot's a3m folder
    prefix = f"uniclust30/{uniprot_id}/a3m/"

    # 3) List objects under that prefix
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents", [])
    if not contents:
        logging.warning(f"No objects found under s3://{bucket}/{prefix}")
        return []

    # 4) Filter for .a3m and download each
    downloaded = []
    for obj in contents:
        key = obj["Key"]
        if key.lower().endswith(".a3m"):
            local_path = os.path.join(local_root, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                s3.download_file(bucket, key, local_path)
                downloaded.append(os.path.abspath(local_path))
                logging.info(f"Downloaded {key}")
            except Exception as e:
                logging.error(f"Failed to download {key}: {e}")

    if not downloaded:
        logging.warning(f"No .a3m files downloaded for uniprot {uniprot_id}")
    else:
        logging.info(f"Downloaded {len(downloaded)} .a3m files for uniprot {uniprot_id}")

    return downloaded

def find_and_download_msa_files(
    info_file_pattern: str,
):
    downloaded_count = 0
    info_files = glob.glob(info_file_pattern)
    if not info_files:
        logging.warning(f"No info files found matching: {info_file_pattern}")

    for info_file in info_files:
        logging.info(f"Reading info file: {info_file}")
        with open(info_file, "r") as f:
            for line in f:
                uniprot_id = line.strip()
                if not uniprot_id:
                    continue

                # Download MSAs for this uniprot ID
                msa_files = download_msas_for_uniprot(uniprot_id)
                if msa_files:
                    downloaded_count += len(msa_files)
                    logging.info(f"Downloaded {len(msa_files)} MSAs for {uniprot_id}")
                else:
                    logging.warning(f"No MSAs found for {uniprot_id}")

    logging.info(f"Total MSAs downloaded: {downloaded_count}")

    return downloaded_count

def run_phylogenetic_inference(msa_path, software='iqtree'):
    """
    Runs phylogenetic inference software on a given MSA file.

    Args:
        msa_path (str): The absolute path to the MSA file.
        software (str): The phylogenetic software to use ('iqtree', 'fasttree', 'raxml-ng').
                        Defaults to 'iqtree'.

    Returns:
        bool: True if the process completed successfully (return code 0), False otherwise.
    """
    if not os.path.exists(msa_path):
        logging.error(f"MSA file does not exist: {msa_path}")
        return False

    # Define output prefix based on MSA filename (saved in the same directory)
    output_dir = os.path.dirname(msa_path)
    base_name = os.path.splitext(os.path.basename(msa_path))[0]
    output_prefix = os.path.join(output_dir, base_name)

    # --- Software Selection Discussion ---
    # FastTree (Double Precision): Very fast, uses approximate Maximum Likelihood (ML).
    #   Pros: Speed, suitable for very large datasets or quick analyses.
    #   Cons: Generally less accurate than full ML methods, fewer options.
    #   Command: FastTreeMP -double -nt -gtr -gamma < msa_path > output_prefix.tree
    #
    # IQ-TREE: Sophisticated ML method.
    #   Pros: High accuracy, automatic model selection (ModelFinder), various analyses (e.g., bootstrap), good parallelization.
    #   Cons: Can be computationally intensive.
    #   Command: iqtree -s msa_path -m MFP -B 1000 -T AUTO -pre output_prefix
    #     -s: input alignment
    #     -m MFP: ModelFinder Plus (find best model and use it)
    #     -B 1000: Ultrafast bootstrap with 1000 replicates
    #     -T AUTO: Automatic thread detection
    #     -pre: Output prefix for all generated files (.treefile, .log, etc.)
    #
    # RAxML-ng: Modern ML method, successor to RAxML.
    #   Pros: High accuracy, speed (often faster than IQ-TREE for simple ML), efficient memory usage.
    #   Cons: Model selection might require separate steps or tools compared to IQ-TREE's ModelFinder.
    #   Command: raxml-ng --msa msa_path --model GTR+G --prefix output_prefix --threads auto --bs-trees 100
    #     --msa: input alignment
    #     --model: Substitution model (e.g., GTR+G, or use auto-prot)
    #     --prefix: Output prefix
    #     --threads: auto
    #     --bs-trees: Number of bootstrap trees (e.g., 100 or more)
    #
    # Choice: IQ-TREE is chosen here for its balance of accuracy, ease of use
    # (automatic model selection), and comprehensive features. Ensure it's installed
    # and accessible in your system's PATH.
    # ---

    command = []
    if software == 'iqtree3':
        # Ensure the output tree file doesn't already exist to avoid re-computation
        # IQ-TREE might handle this, but an explicit check can save time.
        tree_file = f"{output_prefix}.treefile"
        if os.path.exists(tree_file):
            logging.info(f"Skipping: Output tree file already exists: {tree_file}")
            return True # Treat as success if already done

        logging.info(f"Running IQ-TREE on: {msa_path}")
        command = [
            'iqtree3',
            '-s', msa_path,       # Input MSA
            '-m', 'MFP',          # ModelFinder Plus: Auto-select best model + estimate tree
            '-B', '1000',         # 1000 ultrafast bootstrap replicates
            '-T', 'AUTO',         # Auto-detect optimal number of threads
            '--prefix', output_prefix # Set output prefix (basename in same dir)
            # '-nt AUTO' is deprecated in IQ-TREE 2, use -T AUTO
        ]
    elif software == 'fasttree':
         # Example for FastTree Double Precision (adjust model as needed)
        tree_file = f"{output_prefix}.tree"
        if os.path.exists(tree_file):
             logging.info(f"Skipping: Output tree file already exists: {tree_file}")
             return True

        logging.info(f"Running FastTreeMP (Double Precision) on: {msa_path}")
        # Note: FastTree typically reads from stdin and writes to stdout
        # Using shell=True here for redirection, be cautious with untrusted input.
        # A safer approach involves direct piping with subprocess Popen.
        # This simplified version uses shell redirection.
        command_str = f"FastTreeMP -double -nt -gtr -gamma < {msa_path} > {tree_file}"
        logging.info(f"Executing: {command_str}")
        try:
            # Use shell=True carefully for redirection
            result = subprocess.run(command_str, shell=True, check=True, capture_output=True, text=True)
            logging.info(f"FastTree Output for {base_name}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            return True # Assumes success if check=True doesn't raise error
        except subprocess.CalledProcessError as e:
            logging.error(f"FastTree failed for {msa_path} with return code {e.returncode}")
            logging.error(f"STDERR:\n{e.stderr}")
            logging.error(f"STDOUT:\n{e.stdout}")
            # Clean up potentially empty/incomplete output file
            if os.path.exists(tree_file) and os.path.getsize(tree_file) == 0:
                os.remove(tree_file)
            return False
        except FileNotFoundError:
            logging.error(f"FastTreeMP command not found. Is it installed and in PATH?")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred running FastTree: {e}")
            return False

    elif software == 'raxml-ng':
        # Example for RAxML-ng (adjust model as needed, e.g., --model LG+G4+F)
        # RAxML-ng automatically creates output files based on prefix. Check for final tree file.
        tree_file = f"{output_prefix}.raxml.bestTree"
        if os.path.exists(tree_file):
             logging.info(f"Skipping: Output tree file already exists: {tree_file}")
             return True

        logging.info(f"Running RAxML-ng on: {msa_path}")
        command = [
            'raxml-ng',
            '--msa', msa_path,
            '--model', 'LG+G4+F', # Example model for proteins, adjust as needed or use auto
            '--prefix', output_prefix,
            '--threads', 'auto', # Use available cores
            '--bs-trees', '100'  # Number of bootstrap replicates (adjust as needed)
        ]
    else:
        logging.error(f"Unsupported phylogenetic software specified: {software}")
        return False

    # Execute the command for IQ-TREE or RAxML-ng (FastTree handled separately above)
    if command:
        logging.info(f"Executing command: {' '.join(command)}")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Successfully processed {base_name}.")
            # Log stdout/stderr for debugging if needed, can be verbose
            # logging.debug(f"STDOUT:\n{result.stdout}")
            # logging.debug(f"STDERR:\n{result.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Software '{software}' failed for {msa_path} with return code {e.returncode}")
            logging.error(f"STDERR:\n{e.stderr}")
            logging.error(f"STDOUT:\n{e.stdout}")
            return False
        except FileNotFoundError:
            logging.error(f"Command '{command[0]}' not found. Is {software} installed and in PATH?")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred running {software}: {e}")
            return False

    return False # Should not be reached if software is supported

def preprocess_a3m_headers(original_msa_path):
    """
    Reads an A3M file, simplifies headers, and saves to a new file.

    Headers like '>tr|ID1|ID2_species ...' are simplified to '>ID1'.

    Args:
        original_msa_path (str): Path to the original .a3m file.

    Returns:
        str: Path to the new file with simplified headers (e.g., 'original_RENAMED.a3m'),
             or None if an error occurs or the input file doesn't exist.
    """
    if not os.path.exists(original_msa_path):
        logging.error(f"Preprocessing skipped: Input file not found: {original_msa_path}")
        return None

    base, ext = os.path.splitext(original_msa_path)
    renamed_msa_path = f"{base}_RENAMED{ext}"

    # Optional: Skip if renamed file already exists
    if os.path.exists(renamed_msa_path):
        logging.info(f"Preprocessing skipped: Renamed file already exists: {renamed_msa_path}")
        return renamed_msa_path

    logging.info(f"Preprocessing: Simplifying headers for {os.path.basename(original_msa_path)}")
    seen_ids = set() # To check for potential duplicate IDs after simplification
    successful_processing = False
    try:
        with open(original_msa_path, 'r') as infile, open(renamed_msa_path, 'w') as outfile:
            for line in infile:
                if line.startswith('>>'): # Skip consensus lines if present
                    logging.warning(f"Skipping consensus line found: {line.strip()}")
                    # Skip the next line too (the consensus sequence)
                    next(infile)
                    continue
                elif line.startswith('>'):
                    parts = line.strip().split('|')
                    # Expecting format like >db|ID1|ID2_rest...
                    if len(parts) >= 3:
                        simple_id = parts[1] # Use the ID between the first and second pipe

                        # Handle potential duplicate IDs after simplification
                        original_simple_id = simple_id
                        counter = 1
                        while simple_id in seen_ids:
                            logging.warning(f"Duplicate simplified ID '{original_simple_id}' found in {os.path.basename(original_msa_path)}. Appending counter.")
                            simple_id = f"{original_simple_id}_{counter}"
                            counter += 1

                        seen_ids.add(simple_id)
                        outfile.write(f">{simple_id}\n")
                    else:
                        # Could not parse header as expected, write original with warning
                        logging.warning(f"Could not parse header, writing original: {line.strip()}")
                        outfile.write(line)
                else:
                    # It's a sequence line, write it as is
                    outfile.write(line)
            successful_processing = True
    except StopIteration:
         logging.error(f"Error processing {original_msa_path}: Encountered unexpected end of file after '>>' line.")
         successful_processing = False # Mark as failed
    except IOError as e:
        logging.error(f"I/O error processing {original_msa_path}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing {original_msa_path}: {e}")

    if successful_processing:
        logging.info(f"Preprocessing successful. Renamed file saved to: {renamed_msa_path}")
        return renamed_msa_path
    else:
        # Clean up potentially incomplete renamed file if an error occurred
        if os.path.exists(renamed_msa_path):
            try:
                os.remove(renamed_msa_path)
                logging.info(f"Removed incomplete renamed file: {renamed_msa_path}")
            except OSError as e:
                logging.error(f"Could not remove incomplete renamed file {renamed_msa_path}: {e}")
        return None

# --- MAFFT Reformatting Function ---
def reformat_a3m_to_fasta(input_a3m_path, output_fasta_path=None):
    """
    Uses MAFFT command-line tool to reformat an A3M file to standard FASTA.

    Ensures all sequences are uppercase and padded to the same length.

    Args:
        input_a3m_path (str): Path to the input .a3m file (ideally one with
                              already simplified headers).
        output_fasta_path (str, optional): Path for the output .fasta file.
                                           If None, it defaults to replacing
                                           '.a3m' or '_RENAMED.a3m' with '_final.fasta'.

    Returns:
        str: Path to the new .fasta file, or None if an error occurs.
    """
    if not os.path.exists(input_a3m_path):
        logging.error(f"Reformatting skipped: Input file not found: {input_a3m_path}")
        return None

    # Determine output path if not provided
    if output_fasta_path is None:
        if input_a3m_path.endswith("_RENAMED.a3m"):
             base = input_a3m_path[:-len("_RENAMED.a3m")]
             output_fasta_path = f"{base}_final.fasta"
        else:
             base, _ = os.path.splitext(input_a3m_path)
             output_fasta_path = f"{base}_final.fasta"


    # Optional: Skip if final FASTA file already exists
    if os.path.exists(output_fasta_path):
        logging.info(f"Reformatting skipped: Output FASTA file already exists: {output_fasta_path}")
        return output_fasta_path

    # Define the MAFFT command
    # Uses --anysymbol for protein/mixed case, --quiet to reduce console output
    # Quotes are important for paths that might contain spaces
    reformat_command = f"mafft --anysymbol --quiet \"{input_a3m_path}\" > \"{output_fasta_path}\""

    logging.info(f"Reformatting to FASTA using MAFFT: {os.path.basename(input_a3m_path)} -> {os.path.basename(output_fasta_path)}")
    logging.debug(f"Executing command: {reformat_command}") # Log command at debug level

    try:
        # Execute the command using shell for redirection '>'
        result = subprocess.run(reformat_command, shell=True, check=True, capture_output=True, text=True)
        # check=True will raise CalledProcessError if mafft fails (non-zero exit code)

        # Check if output file was actually created and is not empty
        if not os.path.exists(output_fasta_path) or os.path.getsize(output_fasta_path) == 0:
             logging.error(f"MAFFT command seemed to succeed, but output file is missing or empty: {output_fasta_path}")
             logging.error(f"MAFFT STDERR:\n{result.stderr}")
             # Attempt to clean up empty file
             if os.path.exists(output_fasta_path):
                 try: os.remove(output_fasta_path)
                 except OSError: pass
             return None

        logging.info(f"Successfully reformatted. FASTA file saved to: {output_fasta_path}")
        return output_fasta_path

    except subprocess.CalledProcessError as e:
        logging.error(f"MAFFT reformatting failed for {input_a3m_path} with return code {e.returncode}")
        logging.error(f"Command executed: {reformat_command}")
        logging.error(f"STDERR:\n{e.stderr}")
        # Clean up potentially empty/incomplete output file
        if os.path.exists(output_fasta_path):
            try: os.remove(output_fasta_path)
            except OSError: pass
        return None
    except FileNotFoundError:
         # This error occurs if the 'mafft' command itself is not found
         logging.error(f"MAFFT command not found. Is MAFFT installed and in your system's PATH?")
         return None
    except Exception as e:
         logging.error(f"An unexpected error occurred during MAFFT reformatting: {e}")
         # Clean up potentially empty/incomplete output file
         if os.path.exists(output_fasta_path):
             try: os.remove(output_fasta_path)
             except OSError: pass
         return None

# --- Main Execution ---
if __name__ == "__main__":
    # Define the pattern for the input info files
    # IMPORTANT: Update this path to the correct location on your system
    info_file_pattern = "/Users/richardzhu/AM220_Final_Project/Phyla/phyla/dataset_info/openfold_eval_*.txt"

    # Choose the phylogenetic software to use ('iqtree', 'fasttree', 'raxml-ng')
    # Ensure the chosen software is installed and in your PATH
    selected_software = 'iqtree3'
    msas_downloaded = True #if MSAs already downloaded, set to True to skip download step

    logging.info("Starting phylogenetic tree construction process.")
    logging.info(f"Using software: {selected_software}")

    # 1. Find all MSA files
    if not msas_downloaded:
        msa_download_count = find_and_download_msa_files(info_file_pattern)

        if msa_download_count == 0:
            logging.warning("No MSA files found to process. Exiting.")
            exit()

        logging.info(f"Found {len(msa_download_count)} MSA files to process.")

    # 2. Run phylogenetic inference for each MSA
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0 # Count runs skipped because output exists

    # Define the root directory where MSAs are stored after download
    msa_root_dir = "/Users/richardzhu/AM220_Final_Project/msas/uniprot/uniclust30"

    # Use glob to find all .a3m files within the expected structure
    # Pattern: /path/to/msas/uniprot/uniclust30/<uniprot_id>/a3m/*.a3m
    msa_file_pattern = os.path.join(msa_root_dir, "*", "a3m", "uniclust30.a3m")
    logging.info(f"Searching for MSA files using pattern: {msa_file_pattern}")
    msa_file_paths = glob.glob(msa_file_pattern)

    # Convert to absolute paths for consistency
    msa_file_paths = [os.path.abspath(p) for p in msa_file_paths]

    if not msa_file_paths:
        logging.warning(f"No .a3m files found in the expected location: {msa_root_dir}")
        # Depending on desired behavior, you might want to exit here
        # exit()
    else:
        logging.info(f"Found {len(msa_file_paths)} .a3m files to process for phylogenetic inference.")

    # Initialize counters before the loop
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0

    for i, msa_path in enumerate(msa_file_paths[:1]):
        logging.info(f"--- Processing MSA {i+1}/{len(msa_file_paths)}: {os.path.basename(msa_path)} ---")

        # Check for existing output before calling the function (optional, but good for clarity)
        output_dir = os.path.dirname(msa_path)
        base_name = os.path.splitext(os.path.basename(msa_path))[0]
        output_prefix = os.path.join(output_dir, base_name)
        output_exists = False
        if selected_software == 'iqtree':
            output_exists = os.path.exists(f"{output_prefix}.treefile")
        elif selected_software == 'fasttree':
             output_exists = os.path.exists(f"{output_prefix}.tree")
        elif selected_software == 'raxml-ng':
             output_exists = os.path.exists(f"{output_prefix}.raxml.bestTree")

        if output_exists:
            logging.info(f"Skipping: Output already exists for {os.path.basename(msa_path)}")
            skipped_runs += 1
            continue # Skip to the next file

        # # --- Preprocess the MSA file (simplify headers) ---
        # renamed_msa_path = preprocess_a3m_headers(msa_path)

        # if renamed_msa_path is None:
        #     logging.error(f"Failed to preprocess {os.path.basename(msa_path)}. Skipping phylogenetic analysis.")
        #     failed_runs += 1 # Count as failed overall
        #     continue # Skip to next file

        # --- Reformat the MSA file to FASTA ---
        final_fasta_path = reformat_a3m_to_fasta(msa_path) # Pass the renamed path here
        if final_fasta_path is None:
            logging.error(f"Failed to reformat {os.path.basename(msa_path)} to FASTA. Skipping.")
            failed_runs += 1
            continue

        # Run the inference
        success = run_phylogenetic_inference(final_fasta_path, software=selected_software)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
        logging.info(f"--- Finished processing {os.path.basename(msa_path)} ---")


    logging.info("=" * 30)
    logging.info("Phylogenetic tree construction complete.")
    logging.info(f"Total MSAs found: {len(msa_file_paths)}")
    logging.info(f"Successfully processed: {successful_runs}")
    logging.info(f"Skipped (output existed): {skipped_runs}")
    logging.info(f"Failed: {failed_runs}")
    logging.info("=" * 30)
