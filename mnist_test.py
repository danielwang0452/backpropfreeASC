# test vanilla forward gradient on MNIST following "Gradients without Backpropagation"
import copy
import torch
import torch.nn.functional as F
import torch.func as fc
import torch.nn as nn
import torchvision
from torchvision import transforms
#from network import MLP_backprop, jvp_MLP, BackpropLayer
from network_v2 import jvp_MLP
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
                out_size=10, hidden_size=[128, 128, 128], method=method)
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
                # test jvp implementation: setting backprop grads as guesses
                # should yield a jvp value that is the same as ||backprop grads||^2\
                '''
                with torch.enable_grad():
                    
                    in_size = torch.prod(torch.tensor(train_dataset[0][0].shape))
                    layer = nn.Linear(in_size, 10)
                    loss_fn = nn.CrossEntropyLoss(reduction='none')
                    output_gradients = {}
                    weight = nn.Parameter(layer.weight)
                    bias = nn.Parameter(layer.bias)
                    x = nn.Parameter(x)

                    def save_gradient_hook(module, grad_input, grad_output):
                        output_gradients[module] = (grad_input[0], grad_output[0])

                    layer.register_backward_hook(save_gradient_hook)
                    def func(x):
                        return loss_fn(layer(x), y)
                    def func2(weight, bias):
                        return F.cross_entropy(F.linear(x, weight, bias), y, reduction='none')

                    loss = func(x).mean()
                    loss.backward()

                    out, jvp = fc.jvp(func2, (weight, bias), (layer.weight.grad, layer.bias.grad))
                    print(jvp, layer.bias.grad.norm()**2+layer.weight.grad.norm()**2)


                    optimizer.zero_grad(set_to_none=True)
                    out = model.forward(x)
                    loss = model.net[-1](out, y).mean()
                    loss.backward()
                    target_jvp = 0
                    for layer, grad in model.output_gradients.items():
                        if layer.last_layer==True:
                            layer.bias_guess = layer.bias.grad
                            target_jvp += (layer.bias.grad**2).sum()
                            layer.weight_guess = layer.weight.grad
                            target_jvp += (layer.weight.grad ** 2).sum()
                        else:
                            layer.bias_guess = layer.bias.grad
                            target_jvp += (layer.bias.grad ** 2).sum()
                            layer.s_guess = grad
                            target_jvp += (grad ** 2).sum()
                    #print(target_jvp)
                '''
                optimizer.zero_grad()
                loss, jvp = model.jvp_forward(x, y.to(device))
                #print(jvp.mean())
                train_losses.append(loss.mean().item())
                model.set_grad(jvp)
                # Optimizer step
                optimizer.step()

            test = True
            if test:
                test_losses, accuracy = test_act_perturb(model, train_dataloader)
                #wandb.log({"train_loss": np.array(train_losses).mean() ,
                #           "test_loss": np.array(test_losses).mean(),
                #           "test_accuracy": np.array(accuracy).mean()
                #           })
                print(epoch, np.array(train_losses).mean(),  np.array(accuracy).mean())
            test_losses.append(np.array(test_losses).mean())
            test_accuracies.append(np.array(accuracy).mean())
        #wandb.finish()
        return test_losses, test_accuracies


log = True
if log:
    losses = {}
    accuracies = {}
    n_epochs = 100

    #train_losses, train_accuracies = act_perturb('backprop', n_epochs=n_epochs)
    #losses['backprop train losses'] = [float(loss) for loss in train_losses]
    #accuracies['backprop train accuracies'] = [float(acc) for acc in train_accuracies]

    #train_losses, train_accuracies = act_perturb('weight_perturb', n_epochs=n_epochs)
    #losses['weight_perturb train losses'] = [float(loss) for loss in train_losses]
    #accuracies['weight_perturb train accuracies'] = [float(acc) for acc in train_accuracies]

    #train_losses, train_accuracies = act_perturb('act_perturb', n_epochs=n_epochs)
    #losses['act_perturb train losses'] = [float(loss) for loss in train_losses]
    #accuracies['act_perturb train accuracies'] = [float(acc) for acc in train_accuracies]

    train_losses, train_accuracies = act_perturb('W^T', n_epochs=n_epochs)
    losses['W^T train losses'] = [float(loss) for loss in train_losses]
    accuracies['W^T train accuracies'] = [float(acc) for acc in train_accuracies]

    # Save to json
    with open('train_losses.json', 'w') as f:
        json.dump(losses, f)
    with open('train_accuracies.json', 'w') as g:
        json.dump(accuracies, g)
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