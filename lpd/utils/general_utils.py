import lpd.utils.torch_utils as tu
import numpy as np
import random
import uuid

def seed_all(seed):
    tu.seed_torch(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_uuid():
    u = str(uuid.uuid4())
    return u