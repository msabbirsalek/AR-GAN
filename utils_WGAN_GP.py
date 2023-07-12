#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.autograd as autograd
import torchvision
import numpy as np


# Learning Rate Adjustment

def adjust_lr(optimizer, iteration, init_lr = 1e-4, total_iteration = 200000):
    
    gradient = (float(-init_lr) / total_iteration)
    lr = gradient * iteration + init_lr 

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

# Calculate Gradient Penalty Loss for WGAN-GP

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = alpha.expand(real_samples.size(0), real_samples.size(1), real_samples.size(2), real_samples.size(3))
    alpha = alpha.to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = fake.to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


# Denormalize image tensors

def denorm(img_tensors):
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return img_tensors * stats[1][0] + stats[0][0]