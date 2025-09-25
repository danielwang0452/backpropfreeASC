import copy
import torch
import torch.nn.functional as F
import torch.func as fc
import torch.nn as nn
import torchvision
from torchvision import transforms
from network_WT_bias_device import jvp_MLP, device
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import wandb
import numpy as np
import math
import matplotlib.pyplot as plt
import json
from pyhessian import hessian
from mnist1d.data import make_dataset, get_dataset_args
import torch.mps

def compute_gradient(model, optimizer, x, y):
    model.train()
    with torch.enable_grad():
        model.output_gradients = []
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y, reduction='mean')
        loss.backward()
        # model.output_gradients = [ds_1, ds_2, ds_3, d_s_out]
        # for param in model.output_gradients:
        #    print(param.shape)
        act_grads = [param.detach().clone() for param in model.output_gradients][:-1]
        parameters_grads = {}
        for p, param in model.named_parameters():
            parameters_grads[p] = param.grad.detach().clone()
    #parameters_grads = {}
    #for p, param in model.named_parameters():
    #    parameters_grads[p] = param.detach().clone()
    #act_grads = [layer.s_guess for layer in model.linear_layers][:-1]
    return act_grads, parameters_grads

def compute_bias(dL, layer, method):
    assert method in ['weight_perturb', 'act_perturb',  'act_perturb-relu', 'W^T']
    with torch.no_grad():
        if method == 'weight_perturb' or method == 'act_perturb':
            # unbiased
            return torch.tensor(0, dtype=torch.float32).cpu(), torch.prod(torch.tensor(layer.weight.shape, dtype=torch.float32)).sqrt().cpu()
        elif method == 'W^T':
            # for y=mask*(W^Te), cov(y) = mask @ W^TW @ mask where mask is a diagonal matrix
            cov_WT = (layer.W_next.T.detach() @ layer.W_next.detach())  # out, out
            mask = torch.diag_embed(layer.mask.detach()).to(torch.float32)
            cov_y = mask @ cov_WT @ mask  # (B, out, out)
            bias = (dL.unsqueeze(1) @ (cov_y - layer.eye)).squeeze().mean(dim=0)  # (B, out)
            return bias.cpu(), (cov_y**2).sum().sqrt() #torch.linalg.det(cov_y.cpu()).mean()
        elif method == 'act_perturb-relu':
            #cov_WT = layer.W_next.T @ layer.W_next  # out, out
            mask = torch.diag_embed(layer.mask.detach()).to(torch.float32)
            cov_y = mask # (B, out, out)
            bias = (dL.unsqueeze(1) @ (cov_y - layer.eye)).squeeze().mean(dim=0)  # (B, out)
            return bias.cpu(), (cov_y**2).cpu().sum().sqrt() #torch.linalg.det(cov_y.cpu()).mean()

def compute_layer_metrics_test(model, optimizer, x, y, methods=['weight_perturb', 'act_perturb', 'act_perturb-relu', 'W^T']):
    # compute the true gradient
    print('testing compute metrics')
    act_grads, parameters_grads = compute_gradient(model, optimizer, x.to(device), y.to(device))
    # compute bias = (cov(y)-I)dL
    model.eval()
    with torch.no_grad():
        layer_metrics = {}
        for method in methods:
            print(method)
            # compute forward pass to get jvp*guess
            model.method = method
            optimizer.zero_grad()
            loss, jvp = model.jvp_forward(x, y)
            model.set_grad(jvp)

            for l, layer in enumerate(model.linear_layers[:-1]): # omit output layer
            # compute MSE
                if method == 'weight_perturb':
                    dLs = [parameters_grads[f'net.{l * 2}.bias'].flatten(),
                           parameters_grads[f'net.{l * 2}.weight'].flatten()]
                    dL = torch.cat(dLs, dim=0) # (B, out)
                    dL_guess = torch.cat([(layer.bias.grad).detach().clone(), (layer.weight.grad.flatten()).detach().clone()], dim=0)
                else:
                    dL = act_grads[l]  # (B, out)
                    dL_guess = (jvp.unsqueeze(-1)*layer.s_guess).detach()
                bias, cov_frob = compute_bias(dL, layer, method)
                #print((torch.tensor(np.cov(dL_guess, rowvar=False))-cov_WT).mean())
                #print(dL.mean(), dL_guess.mean())
                mse = ((dL_guess - dL)**2).mean(dim=0).cpu()
                var = mse - bias**2
                #var2 = dL_guess.var(dim=0)
                #print((var - var2).mean())
                layer_metrics[f'{method}_layer_{l+1}_mse'] = mse.mean()
                layer_metrics[f'{method}_layer_{l+1}_var'] = var.mean()
                layer_metrics[f'{method}_layer_{l+1}_bias'] = bias.mean()
                layer_metrics[f'{method}_layer_{l+1}_cov_forbenius_norm'] = cov_frob
                #print(mse.mean(), var.mean(), (bias).mean(), var.min())
    return layer_metrics
