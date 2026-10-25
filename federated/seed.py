import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensures deterministic behavior where possible on CPU
    torch.use_deterministic_algorithms(True)

    print(f"Random seed set to {seed}")