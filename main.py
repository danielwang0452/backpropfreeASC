import copy
import torch
import torch.nn.functional as F
import torch.func as fc
import torch.nn as nn
import torchvision
from torchvision import transforms
from network import jvp_MLP
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import wandb
import numpy as np
import math
import matplotlib.pyplot as plt
import json
device = 'cpu'

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
    #wandb.init(
    #
    #    project="forward_gradient",
    #
    #    config={
    #        "learning_rate": 1e-4,
    #        "architecture": "MLP_W^T",
    #        "dataset": "MNIST",
    #        "epochs": 10,
    #    }
    #)
    model = jvp_MLP(in_size=torch.prod(torch.tensor(train_dataset[0][0].shape)),
                out_size=10, hidden_size=[128, 128,  128], method=method)
    params_to_optimize = [
        param for name, param in model.named_parameters()
    ]
    # Create the optimizer with the filtered parameters
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-4)
    test_losses = []
    test_accuracies = []
    with torch.no_grad():
        for epoch in range(n_epochs):
            train_losses = []
            for b, batch in enumerate(train_dataloader):
                x, y = batch
                N, C, H, W = x.shape
                x = torch.reshape(x, (N, C*H*W)).to(device)
                if method == 'backprop':
                    with torch.enable_grad():
                        optimizer.zero_grad()
                        loss = F.cross_entropy(model.forward(x), y)
                        train_losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        continue
                optimizer.zero_grad()
                loss, jvp = model.jvp_forward(x, y.to(device))
                #print(jvp.mean())
                train_losses.append(loss.mean().item())
                model.set_grad(jvp)
                # Optimizer step
                optimizer.step()

            test = True
            if test:
                test_loss, accuracy = test_act_perturb(model, train_dataloader)
                #wandb.log({"train_loss": np.array(train_losses).mean() ,
                #           "test_loss": np.array(test_losses).mean(),
                #           "test_accuracy": np.array(accuracy).mean()
                #           })
                print(method, epoch, np.array(test_loss).mean(),  np.array(accuracy).mean())
            test_losses.append(np.array(test_loss).mean())
            test_accuracies.append(np.array(accuracy).mean())
        #wandb.finish()
        return test_losses, test_accuracies


train = False
plot = True

if train:
    losses = {}
    accuracies = {}
    n_epochs = 1000
    methods = ['W^T', 'act_mixing', 'layer_downstream', 'act_perturb', 'weight_perturb', 'backprop']
    methods = ['W^T', 'act_mixing', 'layer_downstream']
    for method in methods:
        train_losses, train_accuracies = act_perturb(method, n_epochs=n_epochs)
        losses[f'{method} train losses'] = [float(loss) for loss in train_losses]
        accuracies[f'{method} train accuracies'] = [float(acc) for acc in train_accuracies]
        with open(f'{method}_losses.json', 'w') as f:
            json.dump(losses, f)
        with open(f'{method}_accuracies.json', 'w') as g:
            json.dump(accuracies, g)

if plot:
    fir = 'results_1'
    losses, accuracies = {}, {}
    methods = ['W^T', 'act_mixing', 'layer_downstream', 'act_perturb', 'weight_perturb', 'backprop']
    for method in methods:
        with open(f'{fir}/{method}_losses.json', 'r') as f:
            dict = json.load(f)
        losses[f'{method} train losses'] = dict[f'{method} train losses']
        with open(f'{fir}/{method}_accuracies.json', 'r') as g:
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
