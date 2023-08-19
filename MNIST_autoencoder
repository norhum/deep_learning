# An autoencoder model (change gray scale to rgb)
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

torch.manual_seed(1)

#gpu support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters and etc
batch_size = 128
num_epochs = 5
learning_rate = 3e-4
input_size = 3
loss_iter = 100
layer = 4

transforms = transforms.Compose([transforms.ToTensor()])
loss_bin = [] #for plotting the loss

#MNIST data
train_dataset = datasets.CIFAR10(root='/train_data', train=True, download=True, transform=transforms)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = datasets.CIFAR10(root='/test_data', train=False, download=True, transform=transforms)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

#model
class CNNAutoEncoder(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(1, 8, 5), #28*28
        nn.ReLU(),
        nn.Conv2d(8, 16, 5), #24*24
        nn.ReLU(),
        nn.Conv2d(16, 32, 5), #20*20
        nn.ReLU(),
        nn.Conv2d(32, 64, 5), #16*16
        nn.ReLU(),
        nn.Conv2d(64, 128, 5), #12*12
        nn.Tanh(),
    )
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 5),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 5),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 5),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, 5),
        nn.ReLU(),
        nn.ConvTranspose2d(8, input_size, 5), #3*28*28
    )

  def forward(self, x):
    latent_space = self.encoder(x) #128*12*12
    output = self.decoder(latent_space)
    return output

class MLE(nn.Module): #multiple layers of autoencoders in parallel
  def __init__(self, input_size, layer):
    super().__init__()
    self.layers = nn.ModuleList([CNNAutoEncoder(input_size) for i in range(layer)])
    self.conv = nn.Conv2d(layer*input_size, input_size, kernel_size=3, stride=1, padding=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = torch.cat([p(x) for p in self.layers], dim=1)
    out = self.sigmoid(self.conv(out))
    return out

model = MLE(input_size, layer).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#train
for epoch in range(num_epochs):
  lc = []
  for batch_iter, (image, label) in enumerate(train_dataloader):

      #forward pass
      image = image.to(device)
      gray_image = image[:,0]*0.3 + image[:,1]*0.59 + image[:,2]*0.11 #lunimosity method
      gray_image = gray_image.unsqueeze(1)
      prediction = model(gray_image)
      loss = criterion(prediction, image)

      #backward pass and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      #loss
      with torch.no_grad():
        lc.append(loss.item())
        if batch_iter % 100 == 0:
          loss = sum(lc)/len(lc)
          loss_bin.append(loss)
          print(f'epoch:{epoch}, {batch_iter}, loss:{loss:.4f}')

#plot the loss
plt.plot(loss_bin)
plt.show()

a, b = next(iter(test_dataloader))
a = a.to(device)
b = a[:,0]*0.3 + a[:,1]*0.59 + a[:,2]*0.11
b = b.unsqueeze(1)
p = model(b)
a = a.transpose(1, 3).transpose(1, 2).cpu().detach().numpy()
b = b.transpose(1, 3).transpose(1, 2).cpu().detach().numpy()
p = p.transpose(1, 3).transpose(1, 2).cpu().detach().numpy()

for i in range(10):
  plt.subplot(2, 5, i+1)
  plt.imshow(a[i], cmap='gray')
plt.show()
for i in range(10):
  plt.subplot(2, 5, i+1)
  plt.imshow(p[i])
plt.show()
#sadly it doesn't represent rgb well
