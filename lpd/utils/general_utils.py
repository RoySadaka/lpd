import lpd.utils.torch_utils as tu
import numpy as np

def seed_all(seed = 42): # BECAUSE ITS THE ANSWER TO LIFE AND THE UNIVERSE
    tu.seed_torch(seed)
    np.random.seed(seed)
