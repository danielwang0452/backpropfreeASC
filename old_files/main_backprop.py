import copy
import torch
import torch.nn.functional as F
import torch.func as fc
import torch.nn as nn
import torchvision
from torchvision import transforms
from network_WT_bias import jvp_MLP
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import wandb
import numpy as np
import math
import matplotlib.pyplot as plt
import json
from pyhessian import hessian
from mnist1d.data import make_dataset, get_dataset_args
device = 'cpu'
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
    losses = []
    accuracy = []
    for batch in dataloader:
        x, y = batch
        N, C, H, W = x.shape
        x = torch.reshape(x, (N, C*H * W)).to(device)
        out = model.forward(x)
        loss = F.cross_entropy(out, y.to(device))
        losses.append(loss.item())

        preds = F.softmax(out, dim=-1).argmax(dim=-1)
        accuracy.append((preds==y.to(device)).sum()/y.shape[0])
    return losses, accuracy

def act_perturb(method, n_epochs):
    wandb.init(

        project="forward_gradient",

        config={
            "learning_rate": 1e-4,
            "architecture": "MLP_W^T",
            "dataset": "CIFAR10" if not use_mnist else "mnist_1d",
            "epochs": 10,
        }
    )
    model = jvp_MLP(in_size=torch.prod(torch.tensor(train_dataset[0][0].shape)),
                out_size=10, hidden_size=[128, 128,  128], method=method)
    params_to_optimize = [
        param for name, param in model.named_parameters()
    ]
    # Create the optimizer with the filtered parameters
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-4)
    test_losses = []
    test_accuracies = []
    layer_metrics = {}
    with torch.no_grad():
        for epoch in range(n_epochs):
            train_losses = []
            for b, batch in enumerate(train_dataloader):
                x, y = batch
                if not use_mnist:
                    N, C, H, W = x.shape
                    x = torch.reshape(x, (N, C*H*W)).to(device)
                '''
                ###
                optimizer.zero_grad()
                loss, jvp = model.jvp_forward(x, y.to(device))

                #print(jvp)
                train_losses.append(loss.mean().item())
                model.set_grad(jvp)
                fwd_grads = {name: param.grad.clone() for name, param in model.named_parameters()}
                #for name, param in model.named_parameters():
                #    print(name, fwd_grads[name])
                # Optimizer step
                #optimizer.step()
                layer_metrics = compute_bias(model, x, y, jvp)
                top_eigenvalues = compute_hessian(model, x, y)
                #layer_metrics['top_eigenvalues'] = top_eigenvalues
                layer_metrics.update(top_eigenvalues)
                layer_metrics.update({"train_loss": loss.sum().item()})
                wandb.log(layer_metrics)
                # backprop
                '''
                with torch.enable_grad():
                    out = model(x)
                    loss = F.cross_entropy(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                top_eigenvalues = compute_hessian(model, x, y)
                layer_metrics['top_eigenvalues'] = top_eigenvalues
                layer_metrics.update(top_eigenvalues)
                layer_metrics.update({"train_loss": loss.item()})
                wandb.log(layer_metrics)


            test = False
            if test:
                test_loss, accuracy = test_act_perturb(model, train_dataloader)
                #wandb.log({"train_loss": np.array(train_losses).mean() ,
                #           "train_accuracy": np.array(accuracy).mean()
                #           })
                print(method, epoch, np.array(test_loss).mean(),  np.array(accuracy).mean())
                test_losses.append(np.array(test_loss).mean())
                test_accuracies.append(np.array(accuracy).mean())
            if epoch % 50 == 0:
                torch.save(model.state_dict(), f'model_checkpoints/W^T_bias_epoch_{epoch}')
        wandb.finish()
        return test_losses, test_accuracies

def compute_bias(model, x, y, jvp):
    # compute the true gradient
    with torch.enable_grad():
        out = model(x)
        loss = F.cross_entropy(out, y, reduction='mean')
        loss.backward()
        # model.output_gradients = [ds_1, ds_2, ds_3, d_s_out]
        #for param in model.output_gradients:
        #    print(param.shape)
        backprop_grads = [param.clone() for param in model.output_gradients][:-1]
        #print(len(backprop_grads))
        # compute bias = (cov(y)-I)dL
    layer_metrics = {}
    for l, layer in enumerate(model.linear_layers[:-1]): # omit output layer
        # for y=mask*(W^Te), cov(y) = mask @ W^TW @ mask where mask is a diagonal matrix
        dL = backprop_grads[l] # (B, out)
        B, out = dL.shape
        cov_WT = layer.W_next.T @ layer.W_next # out, out
        mask = torch.diag_embed(layer.mask).to(torch.float32)
        cov_y = mask @ cov_WT @ mask  # (B, out, out)
        bias = (dL.unsqueeze(1)@(cov_y - torch.eye(cov_y.shape[-1]))).squeeze().mean(dim=0) # (B, out)
        #print(bias)
    # compute MSE
        dL_guess = jvp.unsqueeze(-1)*layer.s_guess
        #print((torch.tensor(np.cov(dL_guess, rowvar=False))-cov_WT).mean())
        bias2 = (dL_guess - dL).mean(dim=0)
        #print(bias, bias2.mean(dim=0))
        mse = ((dL_guess - dL)**2).mean(dim=0)
        var = mse - bias**2
        #var2 = dL_guess.var(dim=0)
        #print((var - var2).mean())
        layer_metrics[f'layer_{l+1}_mse'] = mse.mean()
        layer_metrics[f'layer_{l+1}_var'] = var.mean()
        layer_metrics[f'layer_{l+1}_bias'] = bias.mean()
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
            top_hessian_eigenvalues[f'eigenvalue_{v}'] = value
    return top_hessian_eigenvalues

train = True
plot = False

dir = 'results_W^T_bias'
if train:
    losses = {}
    accuracies = {}
    n_epochs = 100
    method = 'W^T'
    train_losses, train_accuracies = act_perturb(method, n_epochs=n_epochs)
    losses[f'{method} train losses'] = [float(loss) for loss in train_losses]
    accuracies[f'{method} train accuracies'] = [float(acc) for acc in train_accuracies]
    with open(f'{dir}/{method}_losses.json', 'w') as f:
        json.dump(losses, f)
    with open(f'{dir}/{method}_accuracies.json', 'w') as g:
        json.dump(accuracies, g)

if plot:
    losses, accuracies = {}, {}
    methods = ['W^T', 'act_mixing', 'layer_downstream', 'act_perturb', 'weight_perturb', 'backprop']
    for method in methods:
        with open(f'{dir}/{method}_losses.json', 'r') as f:
            dict = json.load(f)
        losses[f'{method} train losses'] = dict[f'{method} train losses']
        with open(f'{dir}/{method}_accuracies.json', 'r') as g:
            dict2 = json.load(g)
        accuracies[f'{method} train accuracies'] = dict2[f'{method} train accuracies']
        final_accuracy = dict2[f'{method} train accuracies'][-1]
        print(f'{method}: {final_accuracy*100:.1f}%')

    # Plotting the losses
    plt.figure(figsize=(12, 6))
    for method, train_losses in losses.items():
        plt.plot(train_losses, label=method)

    plt.title('Training Losses', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)
    plt.show()

    # Plotting the accuracies
    plt.figure(figsize=(12, 6))
    for method, train_accuracies in accuracies.items():
        plt.plot(train_accuracies, label=method)

    plt.title('Training Accuracies', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.show()
