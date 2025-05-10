import os
from pathlib import Path
import os, zipfile
import pandas as pd
from skbio import TreeNode
from ete3 import Tree
import torch.nn.functional as F
from io import StringIO
from PIL import Image, ImageDraw, ImageFont

class HyperparameterProcessing:
    """
    Utility class for processing hyperparameter-related files based on filename patterns.
    """
    @staticmethod
    def export_basenames(directory: str, extension: str, output_file: str) -> None:
        """
        Scans the given directory for files with the specified extension and writes
        their basenames (without extension) to a .txt file in the current working directory.

        :param directory: Path to the directory to scan.
        :param extension: File extension to filter by (e.g., 'txt' or '.txt').
        :param output_file: Name of the output .txt file to create.

        Useful for getting test set from entire dataset
        """
        # Ensure extension starts with a dot
        ext = extension if extension.startswith('.') else f'.{extension}'
        base_path = Path(directory)
        # Collect basenames without extension
        basenames = [file.stem for file in base_path.iterdir()
                     if file.is_file() and file.suffix == ext]
        # Write to output file in cwd
        with open(output_file, 'w') as out:
            for name in basenames:
                out.write(f"{name}\n")

    @staticmethod
    def merge_directories(dir_a: str, dir_b: str, list_file: str, zip_result_name: str) -> list:
        """
        Returns a combined list of file paths:
        - All files in Directory A whose basenames appear in the provided .txt list.
        - All files in Directory B whose basenames do NOT appear in the provided .txt list.
        Then zips those files into zip_result_name, excluding any __MACOSX* or *.DS_Store entries.
        """
        # Read target basenames into a set for fast lookup
        with open(list_file, 'r') as f:
            target_names = set(line.strip() for line in f)

        merged_files = []
        # Add matching files from Directory A
        for file in Path(dir_a).iterdir():
            if file.is_file() and file.stem in target_names:
                merged_files.append(str(file))
        # Add non-matching files from Directory B
        print(f"Found {len(merged_files)} files in {dir_a} matching the list.")
        for file in Path(dir_b).iterdir():
            if file.is_file() and file.stem not in target_names:
                merged_files.append(str(file))

        print(f"There are {len(merged_files)} files in total. Zipping to {zip_result_name} ...")

        # Zip the merged files
        with zipfile.ZipFile(zip_result_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath in merged_files:
                fname = os.path.basename(filepath)
                if fname.startswith("__MACOSX") or fname == ".DS_Store":
                    continue
                zipf.write(filepath, arcname=fname)

        return merged_files
    
    @staticmethod
    def summarize_rf_dist(list_file: str, tsv_file: str) -> tuple:
        """
        Reads target basenames from list_file (.txt), loads the TSV file,
        filters to rows whose index matches those basenames, and computes
        mean and std dev of the 'rf_dist' column. Prints results.

        :param list_file: Path to .txt file listing target row names (one per line).
        :param tsv_file: Path to the TSV file with an 'rf_dist' column.
        :return: (mean, std_dev)
        """
        # Read target names
        with open(list_file, 'r') as f:
            targets = set(line.strip() for line in f if line.strip())

        # Load TSV; assume first column is the row index
        df = pd.read_csv(tsv_file, sep='\t', index_col=0)

        # Ensure 'rf_dist' column exists
        if 'rf_dist' not in df.columns:
            raise KeyError(f"'rf_dist' column not found in {tsv_file}")

        # Filter to only target rows
        subset = df.loc[df.index.intersection(targets), 'rf_dist']

        # Compute statistics
        mean_val = subset.mean()
        std_val  = subset.std()

        # Print a summary
        print(f"{Path(tsv_file).name} → mean(rf_dist) = {mean_val:.4f}, std(rf_dist) = {std_val:.4f}")

        return mean_val, std_val

    def read_tree_size(tre_filepath, img_filepath="visualization.png"):
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
            img_filepath = img_filepath

            # Create a basic tree style
            ts = TreeStyle()
            ts.show_leaf_name = True # Display leaf names

            # Render the tree to the image file
            print(f"Saving tree visualization to: {img_filepath}")
            # Use dpi for higher resolution if desired, e.g., dpi=300
            ete_tree.render(img_filepath, tree_style=ts)

        except Exception as e:
            # Catch potential errors during ete3 processing/rendering
            print(f"Warning: Failed to generate tree visualization using ete3: {e}")
    
    @staticmethod
    def process_extreme_trees(df_a: pd.DataFrame,
                              df_b: pd.DataFrame,
                              dir_a: str,
                              dir_b: str,
                              reference_dir: str,
                              list_file: str,
                              k: int,
                              name_a: str,
                              name_b: str,
                              save_dir: str) -> None:
        """
        For the K smallest and K largest 'rf_dist' in df_a (restricted to names in list_file),
        retrieve the corresponding .tre files from dir_a, dir_b, and reference_dir,
        run read_tree_size on each, and rename their visualizations with suffixes name_a, name_b, or 'Ref'.
        Adds the rf_dist values from df_a and df_b as labels on the images.
        """

        # load target names
        with open(list_file, 'r') as f:
            targets = {line.strip() for line in f if line.strip()}

        # restrict to targets and pick extremes in df_a
        df_sub = df_a.loc[df_a.index.intersection(targets), 'rf_dist']
        if df_sub.empty:
            print("No matching entries in df_a for the given list_file.")
            return
        lowest = df_sub.nsmallest(k).index.tolist()
        highest = df_sub.nlargest(k).index.tolist()
        extremes = lowest + highest

        # prefetch rf_dist values for df_a and df_b
        rf_a = df_a['rf_dist'] if 'rf_dist' in df_a.columns else pd.Series(dtype=float)
        rf_b = df_b['rf_dist'] if 'rf_dist' in df_b.columns else pd.Series(dtype=float)

        for protein in extremes:
            val_a = rf_a.get(protein, None)
            val_b = rf_b.get(protein, None)

            for dirpath, suffix in [
                    (dir_a, name_a),
                    (dir_b, name_b),
                    (reference_dir, 'Ref')]:
                tre_path = os.path.join(dirpath, f"{protein}.tre")
                if not os.path.isfile(tre_path):
                    print(f"Missing tree file: {tre_path}")
                    continue
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    output_img = os.path.join(save_dir, f"{protein}_{suffix}.png")
                    # generate the tree visualization
                    HyperparameterProcessing.read_tree_size(tre_path, img_filepath=output_img)

                    # annotate the saved image with rf_dist
                    img = Image.open(output_img)
                    draw = ImageDraw.Draw(img)
                    label = ""
                    if suffix == name_a and val_a is not None:
                        parts = name_a.split('_')
                        label = f"{' '.join(p.capitalize() for p in parts)} RF dist: {val_a:.4f}"
                        # label = f"{name_a} RF dist: {val_a:.4f}"
                    elif suffix == name_b and val_b is not None:
                        parts = name_b.split('_')
                        label = f"{' '.join(p.capitalize() for p in parts)} RF dist: {val_b:.4f}"
                        # label = f"{name_b} RF dist: {val_b:.4f}"
                    else:
                        label = suffix
                    draw.text((85, 565), label, fill="black")
                    img.save(output_img)

                except Exception as e:
                    print(f"Error processing {tre_path}: {e}")


if __name__ == "__main__":

    test_set = "May_07_hyperparameter_tuning_experiments/hyperparam_expt_test_set.txt"

    # # Define the two base directories
    # base_dir = Path("May_09_structure_experiments_rd2")
    # dir_b = "May_07_hyperparameter_tuning_experiments/iqtree45"
    # # Loop over each subdirectory in the structure experiments folder
    # for subdir in base_dir.iterdir():
    #     if subdir.is_dir():
    #         # Use the subdirectory name as the zip filename
    #         zip_result_name = str(base_dir / f"{subdir.name}.zip")
    #         # Call merge_directories with dir_a=subdir, dir_b fixed, list_file=test_set
    #         HyperparameterProcessing.merge_directories(
    #             str(subdir),
    #             dir_b,
    #             test_set,
    #             zip_result_name
    #         )

    # tsv_result_dir = "May_09_structure_experiments_rd2/structure_expts_rd2_tsvs"
    # # Collect summary stats for each TSV in the result directory
    # results = []
    # for tsv_path in Path(tsv_result_dir).glob("*.tsv"):
    #     mean_val, std_val = HyperparameterProcessing.summarize_rf_dist(test_set, str(tsv_path))
    #     name = tsv_path.stem
    #     if name.endswith("_merged"):
    #         name = name[:-len("_merged")]
    #     results.append({
    #         "name": name,
    #         "average": round(mean_val, 5),
    #         "std deviation": round(std_val, 5)
    #     })

    # # Write out the summary to CSV
    # summary_df = pd.DataFrame(results)
    # summary_df.sort_values(by="average", inplace=True)
    # summary_csv = "May_09_structure_experiments_rd2/structure_expts_rd2_summary_stats.csv"
    # summary_df.to_csv(summary_csv, index=False)
    # print(f"Saved summary stats to {summary_csv}")

    HyperparameterProcessing.process_extreme_trees(
        df_a=pd.read_csv("May_07_hyperparameter_tuning_experiments/hyperparameter_expt_result_tsvs/trees_hyperbolic_esmc_0507_k_5_pca_10_rescaling_True_merged.tsv", sep='\t', index_col=0),
        df_b=pd.read_csv("May_07_hyperparameter_tuning_experiments/hyperparameter_expt_result_tsvs/trees_esmc_euclidean_0507_dist_euclidean_construction_<function fastme at 0x28c8072e0>_merged.tsv", sep='\t', index_col=0),
        dir_a="May_07_hyperparameter_tuning_experiments/trees_hyperbolic_esmc_0507_k_5_pca_10_rescaling_True",
        dir_b="May_07_hyperparameter_tuning_experiments/trees_esmc_euclidean_0507_dist_euclidean_construction_<function fastme at 0x28c8072e0>",
        reference_dir="May_07_hyperparameter_tuning_experiments/iqtree45",
        list_file=test_set,
        k=2,
        name_a="esmc_hyperbolic",
        name_b="esmc_euclidean",
        save_dir="visualizing_best_trees_folder"
    )
