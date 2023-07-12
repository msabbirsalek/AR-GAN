#!/usr/bin/env python
# coding: utf-8



import os

import copy
import time
import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd
from torchvision.utils import save_image
import torchvision
import torchvision.transforms as T
from tqdm.notebook import tqdm

from WGAN_GP import Generator, Discriminator
from utils_WGAN_GP import adjust_lr, compute_gradient_penalty, denorm

random_seed = 42
torch.manual_seed(random_seed);


def train_WGANGP(train_loader, val_loader, netD, netG, inital_epoch):

    # Parameters
    ITERS = 400000
    INPUT_LATENT = 128 
    LAMBDA = 10 # Gradient penalty lambda hyperparameter
    CRITIC_ITERS = 5 # Critic iterations per generator iteration
    
    device_D = 'cuda'
    device_G = 'cuda'
    
    batch_size = 128 
    in_channel = 3
    height = 32
    width = 32
    
    learning_rate = 1e-4
    display_steps = 500
    
    check_point_path = './trained_models/WGAN_GP/model_snapshots.pth'
    
    # set optimizer for generator and discriminator
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    
    print('Number of training batches: {}, Number of validation batches: {}'.format(len(train_loader), len(val_loader)))

    save_losses = []
    dev_disc_costs = []

    if os.path.exists('./trained_models/WGAN_GP/lisa_losses_gp.pickle'):
        with open ('./trained_models/WGAN_GP/lisa_losses_gp.pickle', 'rb') as fp:
            save_losses = pickle.load(fp)

    if os.path.exists('./trained_models/WGAN_GP/dev_disc_costs.pickle'):
        with open ('./trained_models/WGAN_GP/dev_disc_costs.pickle', 'rb') as fp:
            dev_disc_costs = pickle.load(fp)

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1

    one = one.to(device_D)
    mone = mone.to(device_D)       


    # Training
    print('Training starts ...')

    for iteration in range(inital_epoch, ITERS, 1):

        start_time = time.time()

        adjust_lr(optimizerD, iteration, init_lr = learning_rate, total_iteration = ITERS)
        adjust_lr(optimizerG, iteration, init_lr = learning_rate, total_iteration = ITERS)

        d_loss_real = 0
        d_loss_fake = 0

        #for iter_d in range(CRITIC_ITERS):
        for i, (imgs, _) in enumerate(tqdm(train_loader)):

            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():
                p.requires_grad = True

            real_imgs = autograd.Variable(imgs.to(device_D))

            optimizerD.zero_grad()

            # Sample noise as generator input
            z = autograd.Variable(torch.randn(imgs.size(0), INPUT_LATENT,1,1))
            z = z.to(device_G)

            # Generate a batch of images
            fake_imgs = netG(z).cpu()
            fake_imgs = fake_imgs.to(device_D)

            # Real images
            real_validity = netD(real_imgs)
            d_loss_real = real_validity.mean()
            d_loss_real.backward(mone)

            # Fake images
            fake_validity = netD(fake_imgs)
            d_loss_fake = fake_validity.mean()
            d_loss_fake.backward(one)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(netD, real_imgs.data, fake_imgs.data, device_D)
            gradient_penalty.backward()

            # Adversarial loss
            loss_D = d_loss_fake - d_loss_real + LAMBDA * gradient_penalty

            #loss_D.backward()
            optimizerD.step()
            optimizerG.zero_grad()

            del real_validity, fake_validity, fake_imgs, gradient_penalty, real_imgs

            # Train the generator every n_critic iterations

            if (i + 1)% CRITIC_ITERS == 0 or (i + 1) == len(train_loader):

                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation

                # Generate a batch of images    
                fake_imgs = netG(z).cpu()
                fake_imgs = fake_imgs.to(device_D)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = netD(fake_imgs)
                g_loss = fake_validity.mean()
                g_loss.backward(mone)
                loss_G = -g_loss

                #loss_G.backward()
                optimizerG.step()

                del fake_validity


        save_losses.append([loss_D.item(), loss_G.item()])

        if (iteration + 1) % display_steps == 0 or (iteration + 1) == ITERS:

            print('batch {:>3}/{:>3}, D_cost {:.4f}, G_cost {:.4f}\r'.format(iteration + 1, ITERS,loss_D.item(), loss_G.item()))

            with open('./trained_models/WGAN_GP/lisa_losses_gp.pickle', 'wb') as fp:
                pickle.dump(save_losses, fp)

            # snapshots for model
            modelG_copy = copy.deepcopy(netG)
            modelG_copy = modelG_copy.cpu()

            modelG_state_dict = modelG_copy.state_dict() 

            modelD_copy = copy.deepcopy(netD)
            modelD_copy = modelD_copy.cpu()

            modelD_state_dict = modelD_copy.state_dict() 

            torch.save({
                'netG_state_dict': modelG_state_dict,
                'netD_state_dict': modelD_state_dict,
                'epoch': iteration
                }, check_point_path)

            del modelG_copy, modelG_state_dict, modelD_copy, modelD_state_dict

        # save generator model after certain iteration
        if (iteration + 1) % display_steps == 0 :

            g_path = './trained_models/WGAN_GP/G_lisa_gp_' + str(iteration) + '.pth' 

            model_copy = copy.deepcopy(netG)
            model_copy = model_copy.cpu()
            model_state_dict = model_copy.state_dict()
            torch.save(model_state_dict, g_path)

            del model_copy

        # save LISA generated images by generator model every 1000 time

        if (iteration + 1) % display_steps == 0 :

            denorm_fake_imgs = denorm(fake_imgs)
            save_image(denorm_fake_imgs.data, './Generated_imgs/sample_{}.png'.format(iteration), nrow=8)

            costs_avg = 0.0
            disc_count = 0

            # validate GAN model
            with torch.no_grad():
                for images,_ in val_loader:

                    imgs = images.to(device_D)

                    D = netD(imgs)

                    costs_avg += -D.mean().cpu().data.numpy()
                    disc_count += 1

                    del images, imgs

            costs_avg = costs_avg / disc_count

            dev_disc_costs.append(costs_avg)

            with open('./trained_models/WGAN_GP/dev_disc_costs.pickle', 'wb') as fp:
                pickle.dump(dev_disc_costs, fp)

            print('batch {:>3}/{:>3}, validation disc cost : {:.4f}'.format(iteration, ITERS, costs_avg))