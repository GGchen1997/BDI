import functools
from jax.experimental import optimizers
import jax
import jax.config
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True) # for numerical stability, can disable if not an issue
from jax import numpy as jnp
from jax import scipy as sp
import numpy as np

import neural_tangents as nt
from neural_tangents import stax
from jax import random
from utils import *

import argparse
import design_bench
import copy
import time

parser = argparse.ArgumentParser(description="bi-level sequence learning")

parser.add_argument('--mode', choices=['distill', 'grad', 'both'], type=str, default='both')
parser.add_argument('--task', choices=['TFBind8-Exact-v0', 'Superconductor-RandomForest-v0',
                                       'GFP-Transformer-v0', 'UTR-ResNet-v0', 'HopperController-Exact-v0', 
                                       'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0'], type=str, 
                                       default='TFBind8-Exact-v0')
parser.add_argument('--topk', default=128, type=int)
parser.add_argument('--label', default=10.0, type=float)
parser.add_argument('--gamma', default=0.0, type=float)

parser.add_argument('--outer_lr', default=1e-1, type=float)
parser.add_argument('--Tmax', default=200, type=int)
parser.add_argument('--interval', default=200, type=int)

args = parser.parse_args()

#define kernel
init_fn, apply_fn, kernel_fn = stax.serial(stax.Dense(1), stax.Relu(), stax.Dense(1), stax.Relu(),
                                           stax.Dense(1), stax.Relu(), stax.Dense(1), stax.Relu(), stax.Dense(1), stax.Relu(), stax.Dense(1))
KERNEL_FN = functools.partial(kernel_fn, get='ntk')


def distill(args):
    #design task
    task = design_bench.make(args.task)
    #process data
    task_x, task_y, shape0 = process_data(task, args.task)
    load_weights(args.task, task_y, args.gamma)
    #choose candidates
    indexs = np.argsort(task_y.squeeze())
    index = indexs[-args.topk:]
    x_init = copy.deepcopy(task_x[index])
    y_init = args.label*np.ones((x_init.shape[0], 1))
    #overall before evaluation
    max_score, median_score = evaluate_sample(task, x_init, args.task, shape0)
    print("Before  max {} median {}\n".format(max_score, median_score))
    for x_i in range(x_init.shape[0]):
        # define distill data
        params_init = {'x': x_init[x_i].reshape(1, -1), 'y': y_init[x_i].reshape(1, -1)}
        # instance evaluation before
        score_before, _ = evaluate_sample(task, x_init[x_i], args.task, shape0)
        # use the distill data to define optimizer
        opt_state, get_params, update_fn = get_update_functions(params_init, KERNEL_FN, args.outer_lr, mode=args.mode)
        params = get_params(opt_state)
        # define target bench
        x_target_batch = copy.deepcopy(task_x)
        y_target_batch = copy.deepcopy(task_y)
        for i in range(1, args.Tmax + 1):
            # full batch gradient descent
            opt_state, train_loss = update_fn(i, opt_state, params, x_target_batch, y_target_batch)
            params = get_params(opt_state)
            # post evaluation
            if i % args.interval == 0:
                score_after, _ = evaluate_sample(task, params['x'], args.task, shape0)
                print("Data {} train loss {} score before {} score now {}".format(x_i, train_loss, score_before.squeeze(),
                                                                                  score_after.squeeze()))
        # store the updated distilled data
        x_init[x_i] = params['x'].squeeze()
    max_score, median_score = evaluate_sample(task, x_init, args.task, shape0)
    print("After max {} median {}\n".format(max_score, median_score))


if __name__ == "__main__":
    print(args)
    load_d(args.task)
    load_y(args.task)
    distill(args)


