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
import copy

d = None

def load_d(task):
    global d
    d = np.load("npy/" + task + ".npy", allow_pickle=True)

weights = None

def load_weights(task_name, y, gamma):
    global weights
    #if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0','UTR-ResNet-v0']:
    #    index = np.argsort(y, axis=0).squeeze()
    #    anchor = y[index][-10]
    #    tmp = y>=anchor
    #    weights = tmp/np.sum(tmp)
    #elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
    #        'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0']:
    #    tmp = np.exp(gamma*y)
    #    weights = tmp/np.sum(tmp)
    tmp = np.exp(gamma*y)
    weights = tmp/np.sum(tmp)
    print("weights", np.max(weights), np.min(weights))

y_min = None
y_max = None

def load_y(task_name):
    global y_min
    global y_max
    dic2y = np.load("npy/dic2y.npy", allow_pickle=True).item()
    y_min, y_max = dic2y[task_name]

def process_data(task, task_name):
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0','UTR-ResNet-v0']:
        task_x = task.to_logits(task.x)
    elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
            'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0']:
        task_x = copy.deepcopy(task.x)
    task_x = task.normalize_x(task_x)
    shape0 = task_x.shape
    task_x = task_x.reshape(task_x.shape[0], -1)
    task_y = task.normalize_y(task.y)

    return task_x, task_y, shape0

def evaluate_sample(task, x_init, task_name, shape0):
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0','UTR-ResNet-v0']:
        X1 = x_init.reshape(-1, shape0[1], shape0[2])
    elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
            'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0']:
        X1 = x_init
    X1 = task.denormalize_x(X1)
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0','UTR-ResNet-v0']:
        X1 = task.to_integers(X1)
    Y1 = task.predict(X1)
    max_v = (np.max(Y1)-y_min)/(y_max-y_min)
    med_v = (np.median(Y1)-y_min)/(y_max-y_min)
    return max_v, med_v
    #return np.max(Y1), np.median(Y1)
    

def make_loss_fn(kernel_fn, mode="distill"):

    @jax.jit
    def loss_fn_both(x_support, y_support, x_target, y_target, reg=0):
        #use support set to compute target set loss
        y_support = jax.lax.stop_gradient(y_support)
        k_ss = kernel_fn(x_support, x_support)
        k_ts = kernel_fn(x_target, x_support)
        k_ss_reg = (k_ss + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
        pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, sym_pos=True))
        mse_loss1 = 0.5*jnp.sum(weights*(pred - y_target) ** 2)
        #use target set to compute support set loss
        #k_tt = kernel_fn(x_target, x_target)
        k_st = kernel_fn(x_support, x_target)
        #k_tt_reg = (k_tt + jnp.abs(reg) * jnp.trace(k_tt) * jnp.eye(k_tt.shape[0]) / k_tt.shape[0])
        #pred = jnp.dot(k_st, sp.linalg.solve(k_tt_reg, y_target, sym_pos=True))
        #d = np.load("d.npy", allow_pickle=True)
        #pred = jnp.dot(k_st, sp.linalg.solve(k_tt, y_target, sym_pos=True))
        pred = jnp.dot(k_st, d)
        mse_loss2 = 0.5*jnp.mean((pred - y_support) ** 2)
        #merge loss
        mse_loss = mse_loss1 + mse_loss2
        return mse_loss, mse_loss

    @jax.jit
    def loss_fn_distill(x_support, y_support, x_target, y_target, reg=1e-6):
        y_support = jax.lax.stop_gradient(y_support)
        k_ss = kernel_fn(x_support, x_support)
        k_ts = kernel_fn(x_target, x_support)
        k_ss_reg = (k_ss + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / k_ss.shape[0])
        pred = jnp.dot(k_ts, sp.linalg.solve(k_ss_reg, y_support, sym_pos=True))
        mse_loss = 0.5*jnp.sum(weights*(pred - y_target) ** 2)
        return mse_loss, mse_loss

    @jax.jit
    def loss_fn_grad(x_support, y_support, x_target, y_target, reg=1e-6):
        y_support = jax.lax.stop_gradient(y_support)
        #k_tt = kernel_fn(x_target, x_target)
        #k_tt_reg = (k_tt + jnp.abs(reg) * jnp.trace(k_tt) * jnp.eye(k_tt.shape[0]) / k_tt.shape[0])
        k_st = kernel_fn(x_support, x_target)
        #d = sp.linalg.solve(k_tt_reg, y_target, sym_pos=True)
        #d = np.load("d.npy", allow_pickle=True)
        #pred = jnp.dot(k_st, sp.linalg.solve(k_tt, y_target, sym_pos=True))
        pred = jnp.dot(k_st, d)
        mse_loss = 0.5*jnp.mean((pred - y_support) ** 2)
        return mse_loss, mse_loss

    if mode == "both":
        return loss_fn_both
    elif mode == "distill":
        return loss_fn_distill
    elif mode == "grad":
        return loss_fn_grad

def get_update_functions(init_params, kernel_fn, lr, mode="distill"):
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(init_params)
    loss_fn = make_loss_fn(kernel_fn, mode)
    grad_loss = jax.grad(lambda params, x_target, y_target: loss_fn(params['x'],
                                                                       params['y'],
                                                                       x_target,
                                                                       y_target), has_aux=True)
    @jax.jit
    def update_fn(step, opt_state, params, x_target, y_target):
        dparams, aux = grad_loss(params, x_target, y_target)
        return opt_update(step, dparams, opt_state), aux

    return opt_state, get_params, update_fn
