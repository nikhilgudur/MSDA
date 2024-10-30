#!/usr/bin/env python
# coding: utf-8

# In[326]:


import os
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# In[327]:


latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = './samples'


# In[328]:


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

device


# In[329]:


transform = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

f_mnist = FashionMNIST(root='./data', train=True, download=True, transform=transform)


# In[330]:


img, label = f_mnist[0]
print('Label:', label)
print(img[:,10:15,10:15])
torch.min(img), torch.max(img)


# In[331]:


data_loader = torch.utils.data.DataLoader(dataset=f_mnist,
                                          batch_size=batch_size,
                                          shuffle=True)


# In[332]:


discriminator = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)


generator = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)


# In[333]:


D = discriminator.to(device)
G = generator.to(device)


# In[334]:


bce_criterion = nn.BCELoss()
d_optimizer = lambda lr=0.0002: torch.optim.Adam(D.parameters(), lr=lr)
g_optimizer = lambda lr=0.0002: torch.optim.Adam(G.parameters(), lr=lr)


# In[335]:


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# In[336]:


def reset_grad(lr=0.0002):
    d_optimizer(lr).zero_grad()
    g_optimizer(lr).zero_grad()


# In[337]:


import matplotlib.pyplot as plt

img_norm = denorm(img)
plt.imshow(img_norm[0], cmap='gray')
print('Label:', label)


# In[338]:


y = G(torch.randn(2, latent_size).to(device))
gen_imgs = denorm(y.reshape((-1, 28, 28)).detach())


# In[339]:


plt.imshow(gen_imgs[0].cpu(), cmap='gray')


# In[340]:


plt.imshow(gen_imgs[1].cpu(), cmap='gray')


# In[341]:


def train_generator_discriminator(epoch_num, epoch, total_step, criterion, images, lr=0.0001):
  real_labels = torch.ones(batch_size, 1).to(device) * 0.9
  fake_labels = torch.zeros(batch_size, 1).to(device) * 0.1

  if epoch_num % 2 == 0:
      outputs = D(images + 0.1 * torch.randn_like(images))
      d_loss_real = criterion(outputs, real_labels)
      real_score = outputs

      z = torch.randn(batch_size, latent_size).to(device)
      fake_images = G(z)
      outputs = D(fake_images.detach() + 0.1 * torch.randn_like(fake_images))
      d_loss_fake = criterion(outputs, fake_labels)
      fake_score = outputs

      d_loss = d_loss_real + d_loss_fake
      reset_grad()
      d_loss.backward()
      d_optimizer(lr).step()
  else:
      d_loss = torch.tensor(0.0)
      real_score = fake_score = torch.tensor(0.0)

  z = torch.randn(batch_size, latent_size).to(device)
  fake_images = G(z)
  outputs = D(fake_images)
  g_loss = criterion(outputs, real_labels)

  reset_grad()
  g_loss.backward()
  g_optimizer(lr).step()

  if (epoch_num+1) % 200 == 0:
      print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
            .format(epoch, num_epochs, epoch_num+1, total_step, d_loss.item(), g_loss.item(),
                    real_score.mean().item(), fake_score.mean().item()))

  return g_loss, d_loss, real_score, fake_score, fake_images, fake_labels


# In[342]:


if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# In[343]:


total_step = len(data_loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

for epoch in range(num_epochs):
    epoch_d_loss, epoch_g_loss = 0, 0
    epoch_real_score, epoch_fake_score = 0, 0
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)

        g_loss, d_loss, real_score, fake_score, fake_images, _ = train_generator_discriminator(i, epoch, total_step, bce_criterion, images=images)

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        epoch_real_score += real_score.mean().item()
        epoch_fake_score += fake_score.mean().item()

        if epoch == 0 and i == 0:
            real_images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(real_images), os.path.join(sample_dir, 'real_images.png'))

    d_losses.append(epoch_d_loss / total_step)
    g_losses.append(epoch_g_loss / total_step)
    real_scores.append(epoch_real_score / total_step)
    fake_scores.append(epoch_fake_score / total_step)

    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_losses[-1]:.4f}, g_loss: {g_losses[-1]:.4f}, '
        f'D(x): {real_scores[-1]:.2f}, D(G(z)): {fake_scores[-1]:.2f}')


# In[185]:


torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')


# In[ ]:


plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses, label="G")
plt.plot(d_losses, label="D")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Discriminator Real and Fake Scores During Training")
plt.plot(real_scores, label="Real")
plt.plot(fake_scores, label="Fake")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.show()


# In[186]:


from IPython.display import Image

Image('./samples/fake_images-10.png')


# In[187]:


Image('./samples/fake_images-50.png')


# In[188]:


Image('./samples/fake_images-100.png')


# In[189]:


Image('./samples/fake_images-150.png')


# In[190]:


Image('./samples/fake_images-200.png')


# In[191]:


Image('./samples/fake_images-300.png')


# In[161]:


G.load_state_dict(torch.load('G.ckpt'))
D.load_state_dict(torch.load('D.ckpt'))

additional_epochs = 100

new_lr = 0.0001

for epoch in range(num_epochs, num_epochs + additional_epochs):
    epoch_d_loss, epoch_g_loss = 0, 0
    epoch_real_score, epoch_fake_score = 0, 0

    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)

        g_loss, d_loss, real_score, fake_score, fake_images, _ = train_generator_discriminator(i, epoch, total_step, bce_criterion, images=images, lr=new_lr)

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        epoch_real_score += real_score.mean().item()
        epoch_fake_score += fake_score.mean().item()

    d_losses.append(epoch_d_loss / total_step)
    g_losses.append(epoch_g_loss / total_step)
    real_scores.append(epoch_real_score / total_step)
    fake_scores.append(epoch_fake_score / total_step)

    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'retrained_fake_images-{}.png'.format(epoch+1)))

  # Print progress
    print(f'Epoch [{epoch+1}/{num_epochs + additional_epochs}], d_loss: {d_losses[-1]:.4f}, g_loss: {g_losses[-1]:.4f}, '
        f'D(x): {real_scores[-1]:.2f}, D(G(z)): {fake_scores[-1]:.2f}')


# In[164]:


lsgan_sample_dir = './lsgan_samples'
if not os.path.exists(lsgan_sample_dir):
  os.makedirs(lsgan_sample_dir)

D = discriminator.to(device)
G = generator.to(device)
mse_criterion = nn.MSELoss()

total_step = len(data_loader)

num_epochs = 200

for epoch in range(num_epochs):
  for i, (images, _) in enumerate(data_loader):
      images = images.reshape(batch_size, -1).to(device)

      g_loss, d_loss, real_score, fake_score, fake_images, fake_labels = train_generator_discriminator(i, epoch, total_step, mse_criterion, images)

  if (epoch+1) == 1:
      images = images.reshape(images.size(0), 1, 28, 28)
      save_image(denorm(images), os.path.join(lsgan_sample_dir, 'real_images.png'))

  fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
  save_image(denorm(fake_images), os.path.join(lsgan_sample_dir, 'fake_images-{}.png'.format(epoch+1)))

torch.save(G.state_dict(), 'G_lsgan.ckpt')
torch.save(D.state_dict(), 'D_lsgan.ckpt')

