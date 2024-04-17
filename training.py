import sys
import os
import wandb

# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

import torch
import visdom # http://localhost:8097
import numpy as np

from AttentionNetwork import AttentionNetwork
from data_generator import DataGenerator
from hparams import hparams
from utils import generate_attributedict, AdaptiveLRScheduler

hparams_dict = generate_attributedict(hparams)

logto = "wandb" # "visdom" or "wandb"

if logto == "visdom":
    # initialize visdom
    vis = visdom.Visdom()
    # clear visdom
    vis.close(env="main")
elif logto == "wandb":
    # initialize wandb
    wandb.init(project="multi-head-transformer", entity="g-g", config=hparams_dict)
    # clear wandb
    # wandb.finish()
    # initialize visdom
    vis = visdom.Visdom()
    # clear visdom
    vis.close(env="main")
else:
    raise NotImplementedError

# initialize model
model = AttentionNetwork(**hparams_dict.model)
# move to device either cpu or gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    # give a message 
    print("CUDA is available. The model will be trained on GPU.")
else:
    device = torch.device("cpu")
    # raise a warning
    print("Warning: CUDA is not available. The model will be trained on CPU.")

model.to(device)

if logto == "visdom":
    # visualize the initialization
    weights_dict = model.extract_weights()
    model.visualize_weights(vis, weights_dict)
    model.visualize_eigenvalues(vis, weights_dict)

# initialize data generator
data_generator = DataGenerator(
    L=hparams_dict.model.L,
    dx=hparams_dict.model.dx,
    dy=hparams_dict.model.dy,
    **hparams_dict.data
)

# initialize optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=hparams_dict.train.learning_rate,
    momentum=hparams_dict.train.momentum,
)

# initialize learning rate scheduler
scheduler = AdaptiveLRScheduler(
    optimizer=optimizer,
    learning_rate=hparams_dict.train.learning_rate,
    grad_elbow=hparams_dict.train.grad_elbow,
    max_learning_rate=hparams_dict.train.max_learning_rate,
    min_learning_rate=hparams_dict.train.min_learning_rate,
    method=hparams_dict.train.lr_schedule_method,
)

# initialize the loss
loss_fn = torch.nn.MSELoss()

if logto == "visdom":
    # trace the growth of each task's eigenvalues
    for i in range(hparams_dict.model.dy):
        vis.line(X=np.array([0]), Y=np.array([0]), win=f'lbda_U (task {i})', opts=dict(title=f"lbda_U (task {i})"))
        vis.line(X=np.array([0]), Y=np.array([0]), win=f'lbda_W (task {i})', opts=dict(title=f"lbda_W (task {i})"))

    # track the loss
    vis.line(X=np.array([0]), Y=np.array([0]), win='loss', opts=dict(title="Loss"))

# Let's train the model
for epoch in range(hparams_dict.train.num_epochs):
    # generate data
    z, z_q, y_q = data_generator.generate_data()
    # move data to device
    z = z.to(device)
    z_q = z_q.to(device)
    y_q = y_q.to(device)

    # forward pass
    y_hat, _ = model(query=z_q, key=z, value=z)

    # compute loss
    loss = loss_fn(y_hat, y_q)

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # update weights
    optimizer.step()
    # update learning rate
    scheduler.step()

    # visualize eigenvalues and weights
    if epoch % hparams_dict.train.visualize_every_n_epochs == 0:
        weights_dict = model.extract_weights()
        # if logto == "visdom":
        model.visualize_weights(vis, weights_dict)
        model.visualize_eigenvalues(vis, weights_dict)

        if logto == "visdom":
            # update the loss plot
            vis.line(
                X=np.array([scheduler.cumulative_time]), 
                Y=np.array([loss.item()]), 
                win='loss', 
                name='loss',
                update='append')
        elif logto == "wandb":
            # log the loss
            wandb.log({"loss": loss.item(), "learning_rate": scheduler.last_lr})
        else:
            raise NotImplementedError
    
        # visualize the task specific eigenvalues
        for i in range(hparams_dict.model.dy):
            for h in range(hparams_dict.model.num_heads):
                if logto == "visdom":
                    vis.line(
                        X=np.array([scheduler.cumulative_time]), 
                        Y=np.array([weights_dict['ov_effect_weights_diag_y'][h, i].item()]), 
                        win=f'lbda_U (task {i})', 
                        name=f'head {h}', 
                        update='append')
                    vis.line(
                        X=np.array([scheduler.cumulative_time]), 
                        Y=np.array([weights_dict['kq_effect_weights_diag_x_avg'][h, i].item()]),
                        win=f'lbda_W (task {i})', 
                        name=f'head {h}',
                        update='append')
                elif logto == "wandb":
                    wandb.log({
                        f"lbda_U (task {i}), head {h}": weights_dict['ov_effect_weights_diag_y'][h, i].item(),
                        f"lbda_W (task {i}), head {h}": weights_dict['kq_effect_weights_diag_x_avg'][h, i].item(),
                    })
                else:
                    raise NotImplementedError
            # also visualize the average across all heads
            if logto == "visdom":
                vis.line(
                    X=np.array([scheduler.cumulative_time]), 
                    Y=np.array([weights_dict['ov_effect_weights_diag_y'][:, i].mean().item()]), 
                    win=f'lbda_U (task {i})', 
                    name=f'average across heads', 
                    update='append')
                vis.line(
                    X=np.array([scheduler.cumulative_time]), 
                    Y=np.array([weights_dict['kq_effect_weights_diag_x_avg'][:, i].mean().item()]),
                    win=f'lbda_W (task {i})', 
                    name=f'average across heads',
                    update='append')
            elif logto == "wandb":
                wandb.log({
                    f"lbda_U (task {i}), average across heads": weights_dict['ov_effect_weights_diag_y'][:, i].mean().item(),
                    f"lbda_W (task {i}), average across heads": weights_dict['kq_effect_weights_diag_x_avg'][:, i].mean().item(),
                })
            else:
                raise NotImplementedError

    # print loss
    if epoch % hparams_dict.train.print_every_n_epochs == 0:
        print(f"Epoch: {epoch} Loss: {loss.item()} Learning Rate: {scheduler.last_lr}")
        
    
