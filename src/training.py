
import os
import warnings

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import dataset
import emote2pitch

warnings.filterwarnings('ignore')


def train_emote2pitch(epochs:int, trainloader, device, emote2pitch, params, test_samples):
    """Training function for emote2pitch.
    Args:
        epochs (int): number of epochs to train for.
        trainloader (pytorch dataloader) training image pairs 
        device (str): indicator for device (cpu/gpu) to train on 
        emote2pitch (Emote2Pitch) model to be trained
        params (dict): dictionary of parameters:
            lr (float): learning rate
            betas (tuple(float)): beta parameters for adam optimizer
            batch_size (int): batch size used (only passed for logging)
            L1_lambda (float): L1 lambda used to calculate generator loss
        test_samples (list(tuple(image, sample_id))): emotion sample image and id 
    """
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

    with mlflow.start_run(run_name=params['run_name'], experiment_id=params['experiment_id']):
        # Model Setup 
        emote2pitch.to(device)
        emote2pitch.train()

        # Logging Parameters
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch size', params['batch_size'])
        mlflow.log_param('learning rate', params['lr'])
        mlflow.log_param('pairings_file', params['csv'])
        mlflow.set_tags(params['tags'])
        artifact_pth = mlflow.get_artifact_uri()[8:]
        steps = len(trainloader)

        # Running Epochs
        for epoch in range(0, epochs):
            t_epoch = tqdm(trainloader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=steps, unit='batch')
            
            # Running an Epoch
            for i, (emote_img, spectrogram_img) in enumerate(t_epoch):
                # vars to track loss
                G_losses = []
                D_losses  = []
                L1_losses  = []
                fake_gan_losses = []

                # Move data to device
                emote_img = emote_img.to(device)
                spectrogram_img = spectrogram_img.to(device)

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
                
                # Tracking epoch losses
                G_losses.append(G_loss.item())
                D_losses.append(D_loss.item())
                L1_losses.append(L1_loss.item())
                fake_gan_losses.append(fake_gan_loss.item())

                # TQDM Display
                t_epoch.set_description(f"Epoch {epoch+1}")
                t_epoch.set_postfix(
                    G_loss=G_loss.item(),
                    D_loss=D_loss.item(),
                    L1_loss=L1_loss.item(),
                    fake_gan_loss=fake_gan_loss.item()
                )

            # MLFlow Logging (at each epoch)
            mlflow.log_metric("G loss", np.mean(G_losses), step=epoch)
            mlflow.log_metric("D loss", np.mean(D_losses), step=epoch)
            mlflow.log_metric("L1 loss", np.mean(L1_losses), step=epoch)
            mlflow.log_metric("Fake GAN Loss", np.mean(fake_gan_losses), step=epoch)

            # -----------------
            # Sample generation
            # -----------------

            if ((epoch+1) % params['sample_every'] == 0) or (epoch == 0) or (epoch == epochs-1):

                t_gen = tqdm(test_samples, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=len(test_samples), unit=' batch')
                for j, (emote_img, img_id) in enumerate(t_gen):
                    t_gen.set_description(f"Test sampling ")
                    emote_img = emote_img.to(device)
                    with torch.no_grad():
                        img = emote2pitch.G(emote_img)
                        save_pth = os.path.join(artifact_pth, str(epoch+1))
                        if not os.path.isdir(save_pth):
                            os.makedirs(save_pth)
                        np.save(os.path.join(save_pth, img_id), img.cpu().numpy())
                
                # Saving Model 
                torch.save(emote2pitch.state_dict(), os.path.join(artifact_pth, str(epoch+1), 'mdl.pth'))

if __name__ == "__main__":
    

    sample_rates = ['44100Hz', '22050Hz']
    spect_types = ['constant-q', 'chromagram', 'mel']

    for sr in sample_rates:
        for st in spect_types:
            print(f'Training on : {st} @ {sr}\n')

            ## Params

            csv_pth = os.path.join('data', 'pairings', f'FER2{st}-3-splits-{sr}.csv')
            image_size = 256
            num_workers = 0
            batch_size = 1 
            n_epochs = 300

            device = torch.device("cuda:0" if (torch.cuda.is_available() > 0) else "cpu")
    
            tags = {
                "HZ": sr[:-2],
                "splits": 3,
                "spectrogram": st,
            }

            params = {
                    'lr':2e-4,
                    'betas':(0.5,0.999),
                    'batch_size': 1,
                    'L1_lambda': 100.0,
                    'sample_every':15,
                    'experiment_id':0,
                    'run_name': f'E2P-cGAN-{st}-{tags["HZ"]}HZ-{tags["splits"]}split',
                    'tags': tags,
                    'csv': csv_pth 
                }

            ## Data set setup
            transform = transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5)),
                                    ])

            emo2pitch_train_set = dataset.EmotePairingDataset(csv_pth, transform=transform, target_transform=transform)

            dataloader = torch.utils.data.DataLoader(dataset=emo2pitch_train_set,
                                                    batch_size=batch_size,
                                                    shuffle=True)

            e2p = emote2pitch.Emote2Pitch()

            emots = ['happy', 'angry', 'sad', 'surprise']
            root = os.path.join('data','FER', 'test')
            test_samples = []

            for e in emots:
                emote_pth = os.path.join(root, e)
                sample_files_names = os.listdir(emote_pth)[:2]
                for sample_image in sample_files_names:
                    smpl = torch.zeros((1,1,image_size, image_size))
                    x = Image.open(os.path.join(emote_pth, sample_image))
                    smpl[0] = transform(x)

                    test_samples.append((smpl, f'{e}-{sample_image}'))

            # Training 

            train_emote2pitch(n_epochs, dataloader, device, e2p, params, test_samples=test_samples)