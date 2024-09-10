import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Discriminator con strati convoluzionali
class Discriminator(nn.Module): # utilizziamo una rete neurale feedforward
    def __init__(self, img_channels): # img_dim = 784
        super(Discriminator, self).__init__() # ereditiamo dalla classe nn.Module
        self.disc = nn.Sequential( # utilizziamo un modello sequenziale
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),  # Output: N x 32 x 14 x 14 
            nn.LeakyReLU(0.2), # LeakyReLU è utilizzato per prevenire il problema di ReLU morto nei GAN
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: N x 64 x 7 x 7 
            nn.BatchNorm2d(64), # Batch normalization
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: N x 128 x 3 x 3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=0),  # Output: N x 1 x 1 x 1
            nn.Sigmoid(), # Sigmoid per ottenere la probabilità dell'output tra 0 e 1
        )

    def forward(self, x):
        return self.disc(x).view(-1)  # Output scalare per ogni immagine
    
# Generator con strati convoluzionali
class Generator(nn.Module): # utilizziamo una rete neurale feedforward
    def __init__(self, z_dim, img_channels): # z_dim = 64, img_dim = 784 -> 28*28*1
        super(Generator, self).__init__() 
        self.gen = nn.Sequential( 
            # Input: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, 256, kernel_size=7, stride=1, padding=0), # Output: N x 256 x 7 x 7 
            nn.BatchNorm2d(256),
            nn.ReLU(), # Funzione di attivazione
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: N x 128 x 14 x 14
            nn.BatchNorm2d(128), # Batch normalization
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: N x 64 x 28 x 28
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, kernel_size=3, stride=1, padding=1), # Output: N x img_channels x 28 x 28
            nn.Tanh(),  # Output compreso tra -1 e 1
        )

    def forward(self, x):
        return self.gen(x)

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu" # Verifica se è disponibile una GPU
lr = 3e-4 # Learning rate
z_dim = 64  # Dimensione del vettore latente (rumore)
image_channels = 1  # Canali di output per MNIST (grayscale)
batch_size = 128 # Dimensione del batch
num_epochs = 75 # Numero di epoche

disc = Discriminator(image_channels).to(device) # Discriminatore
gen = Generator(z_dim, image_channels).to(device) # Generatore
fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device) # Noise iniziale

# Trasformazioni per normalizzare il dataset
transforms = transforms.Compose([ 
    transforms.ToTensor(), # Converte l'immagine in un tensore
    transforms.Normalize((0.5,), (0.5,)),  # Normalizza i valori tra -1 e 1
])

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True) 
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
opt_disc = optim.Adam(disc.parameters(), lr=lr) # Ottimizzatore per il discriminatore
opt_gen = optim.Adam(gen.parameters(), lr=lr) # Ottimizzatore per il generatore
criterion = nn.BCELoss()  # Loss per il discriminatore

# Tensorboard writers
writer_fake = SummaryWriter(f"runs/DCGAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/DCGAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader): 
        real = real.to(device) # Spostiamo le immagini reali sulla GPU
        batch_size = real.shape[0] # Dimensione del batch

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device) # Generiamo il rumore
        fake = gen(noise) # Generiamo le immagini fake
        disc_real = disc(real).view(-1) # Passiamo le immagini reali al discriminatore
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # Calcoliamo la loss per le immagini reali
        disc_fake = disc(fake.detach()).view(-1) # Passiamo le immagini fake al discriminatore
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # Calcoliamo la loss per le immagini fake
        lossD = (lossD_real + lossD_fake) / 2 # Calcoliamo la loss media per le immagini reali e fake
        disc.zero_grad() # Azzeriamo i gradienti
        lossD.backward() # Backpropagation
        opt_disc.step() # Aggiorniamo i pesi

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).view(-1) # Passiamo le immagini fake al discriminatore
        lossG = criterion(output, torch.ones_like(output)) # Calcoliamo la loss per le immagini fake
        gen.zero_grad() # Azzeriamo i gradienti
        lossG.backward() # Backpropagation
        opt_gen.step() # Aggiorniamo i pesi

        if batch_idx == 0:
            print( # Stampa delle loss
                f"Epoch [{epoch+1}/{num_epochs}] \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad(): # Disabilitiamo il calcolo dei gradienti
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28) # Reshape delle immagini fake
                data = real.reshape(-1, 1, 28, 28) # Reshape delle immagini reali
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True) # Creiamo una griglia delle immagini fake
                img_grid_real = torchvision.utils.make_grid(data, normalize=True) # Creiamo una griglia delle immagini reali

                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step) # Salviamo le immagini fake
                writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step) # Salviamo le immagini reali
                step += 1
