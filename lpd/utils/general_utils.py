import lpd.utils.torch_utils as tu
import numpy as np

def seed_all(seed):
    tu.seed_torch(seed)
    np.random.seed(seed)
