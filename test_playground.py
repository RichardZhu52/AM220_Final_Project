from esm_functions import fastme

import torch

# Example usage
dm = torch.rand(5, 5)
dm = (dm + dm.t()) / 2.0         # symmetric
dm.fill_diagonal_(0.0)          # zero diagonal
labels = [f"taxon{i}" for i in range(5)]

newick_str = fastme(dm, labels)
print(newick_str)