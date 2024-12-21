# test vanilla forward gradient on MNIST following "Gradients without Backpropagation"
import copy
import torch
import torch.nn.functional as F
import torch.func as fc
import torch.nn as nn
import torchvision
from torchvision import transforms
from network import MLP_backprop, jvp_MLP, BackpropLayer
from network2 import jvp_MLP2
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import wandb
import numpy as np
import math

device = 'cpu'

def exponential_lr_decay(step: int, k: float):
    return math.e ** (-step * k)

# load MNIST dataset
# Define the transform to convert the images to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
])

# Download the MNIST training and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

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

def fwd():
    def fwd_grad(params, model, buffers, names, x, y):
        '''
        Args:
            params: Model parameters.
            buffers: Buffers of the model.
            names: Names of the parameters.
            model: A pytorch model.
            x (torch.Tensor): Input tensor for the PyTorch model.
            y (torch.Tensor): Targets.
        '''
        pred = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
        return F.cross_entropy(pred, y)

    model = MLP(in_size=torch.prod(torch.tensor(train_dataset[0][0].shape)),
                out_size=10, hidden_size=[128, 128]).to(device)

     #base_model = copy.deepcopy(model)
    #base_model.to("meta")

    optimizer = torch.optim.SGD(model.parameters(), lr=2e-4)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: exponential_lr_decay(epoch, k=1e-4))

    loss_fn = nn.CrossEntropyLoss()
    named_buffers = dict(model.named_buffers())
    named_params = dict(model.named_parameters())
    names = named_params.keys()
    params = named_params.values()
    with torch.no_grad():
        for epoch in range(10):
            for batch in train_dataloader:
                x, y = batch
                N, C, H, W = x.shape
                x = torch.reshape(x, (N, H*W))

                # Define the function with fixed arguments using partial
                #f = partial(fwd_grad,model=model,names=names,buffers=named_buffers,x=x.to(device),y=y.to(device),)
                v_params = tuple([torch.randn_like(p) for p in params])
                #loss, jvp = model.jvp_forward(x, y, v_params, params, names, named_buffers)

                f = partial(
                    fwd_grad,
                    model=model,
                    names=names,
                    buffers=named_buffers,
                    x=x.to(device),
                    y=y.to(device),
                )

                # Use `f` in the jvp call
                loss, jvp = fc.jvp(f, (tuple(params),), (v_params,))

                #print(loss, jvp)
                for p in params:
                    #print(p[0][0])
                    break
                # Setting gradients
                for v, p in zip(v_params, params):
                    p.grad = v * jvp

                grad_clipping = 0.0
                if grad_clipping > 0:
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        parameters=params, max_norm=grad_clipping, error_if_nonfinite=True
                    )

                # Optimizer step
                optimizer.step()
                scheduler.step()
                # Zero out grads
                optimizer.zero_grad(set_to_none=True)

                #break
            print(epoch, loss, jvp)
        return
# backprop
def backprop():
    wandb.init(
        # set the wandb project where this run will be logged
        project="forward_gradient",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-4,
            "architecture": "MLP_backprop",
            "dataset": "MNIST",
            "epochs": 10,
        }
    )
    model2 = MLP_backprop(in_size=torch.prod(torch.tensor(train_dataset[0][0].shape)),
                out_size=10, hidden_size=[1024, 1024, 1024, 1024, 1024, 1024]).to(device)
    # [1024, 1024, 1024, 1024, 1024, 1024]
    model2.train()
    optim = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    for epoch in range(500):
        model2.train()
        train_losses = []
        test_losses = []
        for batch in train_dataloader:
            x, y = batch
            N, C, H, W = x.shape
            x = torch.reshape(x, (N, C*H*W))

            optim.zero_grad()

            pred = model2(x)
            loss = F.cross_entropy(pred, y)
            train_losses.append(loss.item())
            # Optimizer step
            loss.backward()
            optim.step()
            #print(f'backprop: {loss}')
            optim.zero_grad(set_to_none=True)
            #break
        test_losses, accuracy = test_model(model2, test_dataloader, backprop=True)
        wandb.log({"train_loss": np.array(train_losses).mean(),
                   "test_loss": np.array(test_losses).mean(),
                   "test_accuracy": np.array(accuracy).mean()})
        print(epoch, np.array(train_losses).mean(), np.array(accuracy).mean())
        #break
    return

def test_model(model, dataloader, backprop=False):
    losses = []
    accuracy = []
    for batch in dataloader:
        x, y = batch
        N, C, H, W = x.shape
        x = torch.reshape(x, (N, C*H * W)).to(device)
        if backprop:
            model.eval()
            with torch.no_grad():
                out = model(x)
        else:
            out, jvp_out = model.jvp_forward(x, y.to(device), return_loss=False)
        loss = F.cross_entropy(out, y.to(device))
        losses.append(loss.item())

        preds = F.softmax(out, dim=-1).argmax(dim=-1)
        accuracy.append((preds==y.to(device)).sum()/y.shape[0])
    return losses, accuracy




def guessgrad():
    wandb.init(
        # set the wandb project where this run will be logged
        project="forward_gradient",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-4,
            "architecture": "MLP_W^T",
            "dataset": "MNIST",
            "epochs": 10,
        }
    )
    def fwd_grad(params, model, buffers, names, x, y):
        '''
        Args:
            params: Model parameters.
            buffers: Buffers of the model.
            names: Names of the parameters.
            model: A pytorch model.
            x (torch.Tensor): Input tensor for the PyTorch model.
            y (torch.Tensor): Targets.
        '''
        pred = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
        return F.cross_entropy(pred, y)

    model = jvp_MLP(in_size=torch.prod(torch.tensor(train_dataset[0][0].shape)),
                out_size=10, hidden_size=[1024, 1024, 1024, 1024, 1024, 1024]).to(device)
    # [1024, 1024, 1024, 1024, 1024, 1024]
    backprop_model = MLP_backprop(torch.prod(torch.tensor(train_dataset[0][0].shape)), 10,
                                 hidden_size=[1024, 1024, 1024, 1024, 1024, 1024])
    '''
    # compare cosine similarity of fwd and W^T to backprop
    # copy weights from jvp model to backprop model
    # order is in_W, in_b, (h0_W, h0_b, ... ), out_W, out_b
    param_list = list(backprop_model.parameters())
    model.in_layer.weight = param_list.pop(0)
    model.in_layer.bias = param_list.pop(0)
    model.out_layer.bias = param_list.pop(-1)
    model.out_layer.weight = param_list.pop(-1)
    for layer in model.hidden_layers:
        layer.weight = param_list.pop(0)
        layer.bias = param_list.pop(0)

    for batch in train_dataloader:
        x, y = batch
        N, C, H, W = x.shape
        x = torch.reshape(x, (N, C * H * W)).to(device)

        out = backprop_model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        param_list = list(backprop_model.parameters())
        backprop_grads = []
        for p in param_list:
            backprop_grads.append(p.grad)
        break
    '''
    params_to_optimize = [
        param for name, param in model.named_parameters() if 'x' not in name
    ]
    # Create the optimizer with the filtered parameters
    #optimizer = torch.optim.SGD(params_to_optimize, lr=1e-4)
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-4)

    named_buffers = dict(model.named_buffers())
    named_params = dict(model.named_parameters())
    names = named_params.keys()
    params = named_params.values()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: exponential_lr_decay(epoch, k=1e-4))
    with torch.no_grad():
        for epoch in range(500):
            train_losses = []
            a_list = []
            b_list = []
            jvps = []
            for b, batch in enumerate(train_dataloader):
                x, y = batch
                N, C, H, W = x.shape
                x = torch.reshape(x, (N, C*H*W)).to(device)

                # copy jvp params to backprop params
                if b % 1 == 10:
                    jvp_params = [model.in_layer.weight.clone(),
                                 model.in_layer.bias.clone()]
                    for layer in model.hidden_layers:
                        jvp_params.append(layer.weight.clone())
                        jvp_params.append(layer.bias.clone())
                    jvp_params.append(model.out_layer.weight.clone())
                    jvp_params.append(model.out_layer.bias.clone())
                    state_dict = backprop_model.state_dict()
                    for (name, _), jvp_params in zip(backprop_model.named_parameters(), jvp_params):
                        state_dict[name].copy_(jvp_params)  # Copy new_param into the corresponding state_dict entry
                    backprop_model.load_state_dict(state_dict)
                    for param in backprop_model.parameters():
                        param.grad = None  # Reset the gradient to None
                    with torch.enable_grad():
                        loss2 = F.cross_entropy(backprop_model(x), y)#,  reduction='sum')
                        grad_xs = []
                        grad_ss = []
                        for i in range(int((len(backprop_model.net)-1)/2)):
                            grad_x = torch.autograd.grad(loss2, getattr(backprop_model, f'x_{i+1}'), retain_graph=True)
                            grad_xs.append(grad_x)
                            grad_s = torch.autograd.grad(loss2, getattr(backprop_model, f's_{i + 1}'),
                                                         retain_graph=True)
                            grad_ss.append(grad_s[0])
                        #grad_Wi = torch.autograd.grad(backprop_model.x_1, backprop_model.net[0].weight, grad_outputs=grad_x1, retain_graph=True)
                        torch.save(grad_xs, 'grad_xs.pth')
                        torch.save(grad_ss, 'grad_ss.pth')
                        loss2.backward()
                    param_list = list(backprop_model.parameters())
                    backprop_grads = []
                    for p in param_list:
                        #if len(p.grad.shape) > 1:
                        backprop_grads.append(p.grad.flatten())

                loss, jvp = model.jvp_forward(x, y.to(device))
                #print((model.in_layer.weight_guess - backprop_grads[0]).max())
                #print(jvp, (model.hidden_layers[0].weight_guess - backprop_grads[2]).max())
                #print(f'loss: {loss.item()}')
                jvps.append(jvp.abs().item())
                train_losses.append(loss.item())
                optimizer.zero_grad(set_to_none=True)
                model.set_grad(jvp)

                if b % 1 == 10:
                    # set grads manually
                    '''
                    jvp_grads = [model.in_layer.weight.grad.flatten(),
                                 model.in_layer.bias.grad.flatten()]
                    for layer in model.hidden_layers:
                        jvp_grads.append(layer.weight.grad.flatten())
                        jvp_grads.append(layer.bias.grad.flatten())
                    jvp_grads.append(model.out_layer.weight.grad.flatten())
                    jvp_grads.append(model.out_layer.bias.grad.flatten())
                    grads_copy = backprop_grads.copy()
                    #print(jvp, model.in_layer.weight_guess.norm(), param_list[0].grad.norm())
                    #model.in_layer.weight.grad = backprop_grads.pop(0)
                    d = backprop_grads.pop(0)
                    model.in_layer.bias.grad = backprop_grads.pop(0)
                    for layer in model.hidden_layers:
                        #layer.weight.grad = backprop_grads.pop(0)
                        d = backprop_grads.pop(0)
                        layer.bias.grad = backprop_grads.pop(0)
                    #model.out_layer.weight.grad = backprop_grads.pop(0)
                    d = backprop_grads.pop(0)
                    model.out_layer.bias.grad = backprop_grads.pop(0)
                    '''
                    # compare cosine similarity with backprop grads
                    # order is in_W, in_b, (h0_W, h0_b, ... ), out_W, out_b
                    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                    jvp_grads = [model.in_layer.weight.grad.flatten(),
                                 model.in_layer.bias.grad.flatten()]
                    for layer in model.hidden_layers:
                        jvp_grads.append(layer.weight.grad.flatten())
                        jvp_grads.append(layer.bias.grad.flatten())
                    jvp_grads.append(model.out_layer.weight.grad.flatten())
                    jvp_grads.append(model.out_layer.bias.grad.flatten())
                    jvp_grads = torch.cat(jvp_grads)
                    backprop_grads = torch.cat(backprop_grads)
                    a = cos(backprop_grads.flatten(), jvp_grads).item()
                    #print(backprop_grads.flatten().norm(), jvp*jvp_grads.norm())
                    #print(a)
                    a_list.append(a)

                # Optimizer step
                optimizer.step()
                #scheduler.step()
                #break
            a_avg = np.array(a_list).mean()
            #print(a_avg)
            #break
            test = True
            #if N != 512:
            #    test = False
            if test:
                test_losses, accuracy = test_model(model, train_dataloader)
                wandb.log({"train_loss": np.array(train_losses).mean() ,
                           "test_loss": np.array(test_losses).mean(),
                           "test_accuracy": np.array(accuracy).mean(),
                           "cosine_similarity": a_avg,
                           })
                print(epoch, np.array(train_losses).mean(), jvp,  np.array(accuracy).mean(), a_avg)
        wandb.finish()
        return

#fwd()
#backprop()
guessgrad()