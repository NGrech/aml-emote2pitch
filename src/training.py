
from statistics import mode
from time import sleep

import mlflow
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


def train_emote2pitch(epochs:int, trainloader, device, emote2pitch, params):
    # Pre training setup 
    G_optimizer = optim.Adam(
        emote2pitch.G.parameters(),
        lr=params['lr'],
        betas=params['betas']
    )
    D_optimizer = optim.Adam(
        emote2pitch.D.parameters(),
        lr=params['lr'],
        betas=params['betas']
    )

    bce_criterion = nn.BCELoss()
    L1_criterion = nn.L1Loss()

    with mlflow.start_run():
        # Model Setup 
        emote2pitch.to(device)
        emote2pitch.train()

        # Logging Parameters
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch size', params['batch_size'])
        mlflow.log_param('learning rate', params['lr'])

        # Running Epochs
        for epoch in range(1, epochs+1):
            t_epoch = tqdm(trainloader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=len(trainloader), unit=' batches')
            
            # Running an Epoch
            for i, (emote_img, spectrogram_img) in enumerate(t_epoch):

                # Move data to device
                emote_img.to(device)
                spectrogram_img.to(device)

                # label setup
                b_size = emote_img.shape[0]
                real_class = torch.ones(b_size,1,16,16).to(device)
                fake_class = torch.zeros(b_size,1,16,16).to(device)

                # ----------------------
                # Training Discriminator
                # ----------------------
                emote2pitch.D.zero_grad()

                real_patch = emote2pitch.D(spectrogram_img, emote_img)
                real_gan_loss = bce_criterion(real_patch,real_class)

                fake=emote2pitch.G(emote_img)
  
                fake_patch = emote2pitch.D(fake.detach(), emote_img)
                fake_gan_loss = bce_criterion(fake_patch, fake_class)

                # Discriminator loss
                D_loss = real_gan_loss + fake_gan_loss
                D_loss.backward()
                D_optimizer.step()

                # ------------------
                # Training Generator
                # ------------------
                emote2pitch.G.zero_grad()

                fake_patch = emote2pitch.D(fake,emote_img)
                fake_gan_loss = bce_criterion(fake_patch,real_class)

                L1_loss = L1_criterion(fake,spectrogram_img)
                G_loss = fake_gan_loss + params['L1_lambda']*L1_loss
                G_loss.backward()
                
                G_optimizer.step()
                

                # MLFlow Logging
                # eg: mlflow.log_metric("loss", loss)
                mlflow.log_metric("G loss", G_loss.item())
                mlflow.log_metric("D loss", D_loss.item())
                mlflow.log_metric("L1 loss", L1_loss.item())
                mlflow.log_metric("Fake GAN Loss", fake_gan_loss.item())

                # TQDM Display
                t_epoch.set_description(f"Epoch {epoch}")
                t_epoch.set_postfix(
                    G_loss=G_loss.item(),
                    D_Loss=D_loss.item(),
                    L1_loss=L1_loss.item(),
                    fake_gan_loss=fake_gan_loss.item()
                )

if __name__ == "__main__":
    trainloader = list(range(10))
    epochs = 2


