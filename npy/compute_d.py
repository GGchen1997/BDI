import os
import re
import requests
import numpy as np
import functools

from jax.experimental import optimizers
import jax
import jax.config
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True) # for numerical stability, can disable if not an issue
from jax import numpy as jnp
from jax import scipy as sp

import neural_tangents as nt
from neural_tangents import stax
from jax import random

import argparse
import design_bench
import copy
import time
from utils import *

parser = argparse.ArgumentParser(description="bi-level sequence learning")

parser.add_argument('--task',  choices=['TFBind8-Exact-v0', 'Superconductor-RandomForest-v0',
                                       'GFP-Transformer-v0', 'UTR-ResNet-v0', 'HopperController-Exact-v0',
                                       'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0'], 
                                       type=str, default='UTR-ResNet-v0')
args = parser.parse_args()

init_fn, apply_fn, kernel_fn = stax.serial(stax.Dense(1), stax.Relu(), stax.Dense(1), stax.Relu(),stax.Dense(1), stax.Relu(), stax.Dense(1), stax.Relu(), stax.Dense(1), stax.Relu(), stax.Dense(1))
KERNEL_FN = functools.partial(kernel_fn, get='ntk')

task = design_bench.make(args.task)
print(task.x.shape)
#process data
x_target, y_target, shape0 = process_data(task, args.task)
reg = 1e-6
print("x_target {} y_target {}".format(x_target.shape, y_target.shape))
k_tt = KERNEL_FN(x_target, x_target)
k_tt_reg = (k_tt + jnp.abs(reg) * jnp.trace(k_tt) * jnp.eye(k_tt.shape[0]) / k_tt.shape[0])
d = sp.linalg.solve(k_tt_reg, y_target, sym_pos=True)
np.save("npy/" + args.task + ".npy", d)
