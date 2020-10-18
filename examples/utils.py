import torch as T
import torch.nn as nn
from lpd.extensions.custom_layers import Dense

def examples_data_generator(N, D_in, D_out, binary_out=False):
    # Create random Tensors to hold inputs and outputs
    x = T.randn(N, D_in)
    if binary_out:
        y = T.randint(low=0,high=2, size=[N, D_out], dtype=T.float)
    else:
        y = T.randn(N, D_out)
    while True:
        yield x, y #YIELD THE SAME X,y every time

def examples_prediction_data_generator(origin_data_loader, steps):
    for x,y in origin_data_loader:
        steps -= 1
        yield x
        if steps == 0:
            break


def get_basic_model(D_in, H, D_out):
    return nn.Sequential(
                            Dense(D_in, H, use_bias=True, activation=nn.ReLU()),
                            Dense(H, D_out, use_bias=True, activation=None)
                        )