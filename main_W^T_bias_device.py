import copy
import torch
import torch.nn.functional as F
import torch.func as fc
import torch.nn as nn
import torchvision
from torchvision import transforms
from network_WT_no_bias_device import jvp_MLP, device
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from metrics import compute_layer_metrics_test
import wandb
import numpy as np
import math
import matplotlib.pyplot as plt
import json
from pyhessian import hessian
from mnist1d.data import make_dataset, get_dataset_args
import torch.mps
# mnist 1d
use_mnist = True
if use_mnist:
    defaults = get_dataset_args()
    data = make_dataset(defaults)
    x, y, x_test, y_test = data['x'], data['y'], data['x_test'], data['y_test']
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    tensor_test_x = torch.tensor(x_test, dtype=torch.float32)
    tensor_test_y = torch.tensor(y_test, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    test_dataset = torch.utils.data.TensorDataset(tensor_test_x, tensor_test_y)
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
    ])

    # Download the CIFAR-10 training and test datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

def test_act_perturb(model, dataloader):
    model.eval()
    with torch.no_grad():
        losses = []
        accuracy = []
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            if not use_mnist:
                N, C, H, W = x.shape
                x = torch.reshape(x, (N, C * H * W)).to(device)
            out = model.forward(x.to(device))
            loss = F.cross_entropy(out, y.to(device))
            losses.append(loss.cpu().item())

            preds = F.softmax(out, dim=-1).argmax(dim=-1)
            accuracy.append((preds==y).cpu().sum()/y.shape[0])
    return losses, accuracy

def act_perturb(methods, n_epochs):
    wandb.init(

        project="forward_gradient",

        config={
            "learning_rate": 1e-4,
            "architecture": "MLP_W^T",
            "dataset": "CIFAR10" if not use_mnist else "mnist_1d",
            "epochs": n_epochs,
        }
    )
    model = jvp_MLP(in_size=torch.prod(torch.tensor(train_dataset[0][0].shape)),
                out_size=10, hidden_size=[128, 128, 128], method='W^T')
    model.to(device)
    params_to_optimize = [
        param for name, param in model.named_parameters()
    ]
    # Create the optimizer with the filtered parameters
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-4)
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []
    model.eval()
    with torch.no_grad():
        for epoch in range(n_epochs):
            layer_metrics = {}
            test = True
            if test:
                train_loss, train_accuracy = test_act_perturb(model, train_dataloader)
                test_loss, test_accuracy = test_act_perturb(model, test_dataloader)
                print(method, epoch, np.array(train_loss).mean(), np.array(train_accuracy).mean())
                test_losses.append(np.array(test_loss).mean())
                test_accuracies.append(np.array(test_accuracy).mean())
                train_losses.append(np.array(train_loss).mean())
                train_accuracies.append(np.array(train_accuracy).mean())
            for b, batch in enumerate(train_dataloader):
                torch.mps.empty_cache()
                #print(torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory())
                model.method = 'W^T'
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                if not use_mnist:
                    N, C, H, W = x.shape
                    x = torch.reshape(x, (N, C*H*W)).to(device)
                ###
                optimizer.zero_grad()
                loss, jvp = model.jvp_forward(x.to(device), y.to(device))

                #print(jvp)
                #train_losses.append(loss.mean().item())
                model.set_grad(jvp)
                # Optimizer step
                optimizer.step()
                layer_metrics = compute_layer_metrics(model, optimizer, x, y, methods)
                layer_metrics["train_accuracy"] = np.array(train_accuracy).mean()
                layer_metrics["test_accuracy"] = np.array(test_accuracy).mean()
                wandb.log(layer_metrics)
            #torch.save(model.state_dict(), f'model_checkpoints/W^T')
            #model = None
            #model = jvp_MLP(in_size=torch.prod(torch.tensor(train_dataset[0][0].shape)),
            #                out_size=10, hidden_size=[1024, 1024, 1024], method='W^T')

        wandb.finish()
        return test_losses, test_accuracies

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
            if 'bias' not in p:
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
            return torch.tensor(0, dtype=torch.float32).cpu(), \
                torch.prod(torch.tensor(layer.weight.shape, dtype=torch.float32)).sqrt().cpu(), \
                torch.tensor(layer.weight.shape[0], dtype=torch.float32).cpu()
        elif method == 'W^T':
            # for y=mask*(W^Te), cov(y) = mask @ W^TW @ mask where mask is a diagonal matrix
            cov_WT = (layer.W_next.T.detach() @ layer.W_next.detach())  # out, out
            mask = torch.diag_embed(layer.mask.detach()).to(torch.float32)
            cov_y = mask @ cov_WT @ mask  # (B, out, out)
            bias = (dL.unsqueeze(1) @ (cov_y - layer.eye)).squeeze().mean(dim=0)  # (B, out)
            return bias.cpu(), ((cov_y - layer.eye)**2).mean().sqrt().cpu(), torch.vmap(torch.trace)(cov_y)  #torch.linalg.det(cov_y.cpu()).mean()
        elif method == 'act_perturb-relu':
            #cov_WT = layer.W_next.T @ layer.W_next  # out, out
            mask = torch.diag_embed(layer.mask.detach()).to(torch.float32)
            cov_y = mask # (B, out, out)
            bias = (dL.unsqueeze(1) @ (cov_y - layer.eye)).squeeze().mean(dim=0)  # (B, out)
            return bias.cpu(), ((cov_y - layer.eye)**2).mean().sqrt().cpu(), torch.vmap(torch.trace)(cov_y) #torch.linalg.det(cov_y.cpu()).mean()

def compute_layer_metrics(model, optimizer, x, y, methods=['weight_perturb', 'act_perturb', 'act_perturb-relu', 'W^T']):
    # compute the true gradient
    act_grads, parameters_grads = compute_gradient(model, optimizer, x.to(device), y.to(device))
    # compute bias = (cov(y)-I)dL
    model.eval()
    with torch.no_grad():
        layer_metrics = {}
        for method in methods:
            # compute forward pass to get jvp*guess
            #print(method)
            model.method = method
            optimizer.zero_grad()
            loss, jvp = model.jvp_forward(x, y)
            model.set_grad(jvp)

            for l, layer in enumerate(model.linear_layers[:-1]): # omit output layer
            # compute MSE
                if method == 'weight_perturb':
                    #dLs = [parameters_grads[f'net.{l * 2}.bias'].flatten(),
                    dLs = [parameters_grads[f'net.{l * 2}.weight'].flatten()]
                    dL = torch.cat(dLs, dim=0) # (B, out)
                    #dL_guess = torch.cat([(layer.bias.grad).detach().clone(), (layer.weight.grad.flatten()).detach().clone()], dim=0)
                    dL_guess = layer.weight.grad.flatten().detach().clone()
                else:
                    dL = act_grads[l]  # (B, out)
                    dL_guess = (jvp.unsqueeze(-1)*layer.s_guess).detach()
                bias, cov_frob, cov_trace = compute_bias(dL, layer, method)
                #print((torch.tensor(np.cov(dL_guess, rowvar=False))-cov_WT).mean())
                #print(dL.mean(), dL_guess.mean())
                mse = ((dL_guess - dL)**2).mean(dim=0).cpu()
                var = mse - bias**2
                #var2 = dL_guess.var(dim=0)
                #print((var - var2).mean())
                layer_metrics[f'{method}_layer_{l+1}_mse'] = mse.mean()
                layer_metrics[f'{method}_layer_{l+1}_var_linf'] = var.abs().max()
                layer_metrics[f'{method}_layer_{l+1}_var_linf'] = torch.linalg.norm(var)
                layer_metrics[f'{method}_layer_{l + 1}_cov_trace'] = cov_trace.mean().cpu()
                layer_metrics[f'{method}_layer_{l+1}_bias_linf'] = bias.abs().max()
                layer_metrics[f'{method}_layer_{l+1}_bias_l2'] = torch.linalg.norm(bias)
                layer_metrics[f'{method}_layer_{l+1}_cov_forbenius_norm'] = cov_frob
                #print(mse.mean(), var.mean(), (bias).mean(), var.min())
    return layer_metrics

# compute top k hessian eigenvalues and eigenvectors using pyhessian
def compute_hessian(model, x, y):
    top_hessian_eigenvalues = {}
    with torch.enable_grad():
        hessian_comp = hessian(model, criterion=nn.CrossEntropyLoss(), data=(x, y), cuda=False)
        top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=15)
        #print(top_eigenvalues)
        #print(top_eigenvectors[0])
        for v, value in enumerate(sorted(top_eigenvalues)):
            top_hessian_eigenvalues[f'eigenvalue_/{v}'] = value
    return top_hessian_eigenvalues

train = True
plot = False

dir = 'results_W^T_bias'
if train:
    losses = {}
    accuracies = {}
    n_epochs = 300
    method = 'W^T'
    #train_losses, train_accuracies = act_perturb(['W^T'], n_epochs=n_epochs)
    train_losses, train_accuracies = act_perturb(['W^T'], n_epochs=n_epochs)
