import lpd.utils.torch_utils as tu
import numpy as np
import random

def seed_all(seed):
    tu.seed_torch(seed)
    np.random.seed(seed)
    random.seed(seed)
