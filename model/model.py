from model.sub_models import Discriminator, Generator
import torch
import torch.nn as nn
from model.utils import weights_init

class GAN():
    def __init__(self, ngpu, nz, ngf, ndf, nc, device):

        self.generator = Generator(ngpu, nz, ngf, nc).to(device)
        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            self.generator = nn.DataParallel(self.generator, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.generator.apply(weights_init)

        self.discriminator = Discriminator(ngpu, ndf, nc).to(device)
        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.discriminator.apply(weights_init)

        self.nz = nz
        self.device = device

    def define_loss_and_opt(self, loss_function, opt_D, optG):
        self.loss_function = loss_function

        self.optimizerD = opt_D
        self.optimizerG = optG

    def define_training_params(self, batch_size, real_label, fake_label):
        self.batch_size = batch_size
        self.real_label = real_label
        self.fake_label = fake_label

    def discriminator_step(self, b_data, labels):
        #? Train with all-real batch
        self.discriminator.zero_grad()
        # Forward pass real batch through D
        output = self.discriminator(b_data).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.loss_function(output, labels)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        #? Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)
        # Generate fake image batch with G
        b_fake_data = self.generator(noise)
        labels.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.discriminator(b_fake_data.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.loss_function(output, labels)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()

        return D_x, D_G_z1, errD, b_fake_data

    def generator_step(self, b_fake_data, labels):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.generator.zero_grad()
        labels.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator(b_fake_data).view(-1)
        # Calculate G's loss based on this output
        errG = self.loss_function(output, labels)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()

        return D_G_z2, errG
