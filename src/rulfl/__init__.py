"""Federated RUL prediction under label skew (C-MAPSS FD004)."""
import os

# cuBLAS reads this when CUDA starts; with torch.use_deterministic_algorithms(True) it makes GPU matrix
# products repeatable (M5, M8). Harmless on CPU. setdefault: a value set outside wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
