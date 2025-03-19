import os
import random  # Rastgele örnekleme için
import numpy as np
from PIL import Image, ImageFile
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F_func  # Alias for torch.nn.functional

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

##############################################
# Custom Transforms for Sentinel-2 (Domain B) #
##############################################
class SentinelToTensor(object):
    def __call__(self, x):
        # x: numpy array with shape (H, W, C) where C is expected (e.g., 13)
        # Convert to torch tensor with shape (C, H, W)
        return torch.from_numpy(x).permute(2, 0, 1).float()

class SentinelResize(object):
    def __init__(self, size):
        self.size = size  # size: (height, width)
    def __call__(self, x):
        # x: torch tensor of shape (C, H, W)
        x = F_func.interpolate(x.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False).squeeze(0)
        return x

#######################################
# Channel Attention Module (SE Block) #
#######################################
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

#########################
# ResNet Block for Generator
#########################
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

########################################
# Generator with Channel Attention     
# (For conditional generation, G outputs only extra channels (10))
########################################
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features * 2
        # Residual Blocks
        for _ in range(n_residual_blocks):
            model += [ResnetBlock(in_features)]
        # Channel Attention Module
        model += [ChannelAttention(in_features, reduction=16)]
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True)
            ]
            in_features = out_features
            out_features = in_features // 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),  # output_nc will be 10 for G
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

#########################
# PatchGAN Discriminator#
#########################
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True)
        ]
        model += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

##########################################
# Dataset: Domain A (RGB) & Domain B (S2)
##########################################
class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform_A=None, transform_B=None, sample_percentage=1.0):
        self.files_A = sorted(os.listdir(root_A))
        self.files_B = sorted(os.listdir(root_B))
        # Eğer sentinel2 dosyalarının sadece %10'unu kullanmak istiyorsanız:
        if sample_percentage < 1.0:
            sample_size = max(1, int(len(self.files_A) * sample_percentage))
            self.files_A = random.sample(self.files_A, sample_size)
        self.root_A = root_A
        self.root_B = root_B
        self.transform_A = transform_A
        self.transform_B = transform_B
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    def __getitem__(self, index):
        # Domain A: RGB image
        A_path = os.path.join(self.root_A, self.files_A[index % len(self.files_A)])
        img_A = Image.open(A_path).convert('RGB')
        if self.transform_A:
            img_A = self.transform_A(img_A)
        # Domain B: Sentinel-2 image with error handling
        attempts = 0
        while attempts < len(self.files_B):
            B_path = os.path.join(self.root_B, self.files_B[index % len(self.files_B)])
            try:
                img_B = tiff.imread(B_path)  # Expected shape: (channels, height, width)
                if img_B.ndim == 3 and img_B.shape[0] in [1, 13]:
                    img_B = np.moveaxis(img_B, 0, -1)  # Convert to (height, width, channels)
                if self.transform_B:
                    img_B = self.transform_B(img_B)
                break
            except Exception as e:
                print(f"Skipping file {B_path} due to error: {e}")
                index = (index + 1) % len(self.files_B)
                attempts += 1
        else:
            raise RuntimeError("No valid Sentinel-2 file found in dataset.")
        return {'A': img_A, 'B': img_B}
    
########################################
#          Training Settings           #
########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
n_epochs = 10
lr = 0.0002
lambda_cycle = 10.0
lambda_identity = 5.0

# Transformations for Domain A (RGB)
transform_A = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Custom transformations for Domain B (Sentinel-2: 13 channels)
transform_B = transforms.Compose([
    SentinelToTensor(),
    SentinelResize((256,256)),
    transforms.Normalize((0.5,)*13, (0.5,)*13)
])
input_nc_A = 3   # RGB images
input_nc_B = 13  # Sentinel-2 images (13 channels)

# Folders
band_folder = '/content/drive/MyDrive/data_cycle_gan_sampled/sentinel2_sampled'
root_A = '/content/drive/MyDrive/generated_images'

print("Training is beginning. Target domain:", band_folder)

# Dataset with 10% of Sentinel-2 images selected randomly
dataset = ImageDataset(root_A=root_A, root_B=band_folder, transform_A=transform_A, transform_B=transform_B, sample_percentage=0.1)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
# For mapping A -> B, G produces extra 10 channels.
# Final fake B is built by concatenating input RGB and generated extra channels,
# such that:
# - Index 0: G-generated extra channel (for b1)
# - Indices 1-3: real_A (RGB) → for b2, b3, b4
# - Indices 4-12: G-generated extra channels (for b5 ... b13)
G = Generator(input_nc_A, output_nc=10).to(device)
# For mapping B -> A, F remains unchanged (maps 13 channels to 3 channels)
F = Generator(input_nc_B, input_nc_A).to(device)
    
# Discriminators
D_A = Discriminator(input_nc_A).to(device)
D_B = Discriminator(input_nc_B).to(device)

# Loss Functions
criterion_GAN = nn.MSELoss()    # LSGAN loss
criterion_cycle = nn.L1Loss()   # Cycle consistency loss
criterion_identity = nn.L1Loss()  # Identity loss

# Optimizers
optimizer_G = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)  # RGB image (domain A)
        real_B = batch['B'].to(device)  # Sentinel-2 image (13 channels, domain B)

        # Real/Fake labels
        valid = torch.ones(real_A.size(0), 1, 30, 30, device=device)
        fake = torch.zeros(real_A.size(0), 1, 30, 30, device=device)

        # -----------------
        #  Generator Training
        # -----------------
        optimizer_G.zero_grad()
        # Identity loss is set to 0 in this conditional generation setup
        loss_identity = 0

        # --- Mapping A -> B ---
        fake_extra = G(real_A)  # G generates extra 10 channels from RGB input (real_A), shape: (batch, 10, H, W)
        # Construct final fake_B with preserved channel order:
        # - Index 0: fake_extra[:, 0, :, :] (G-generated for b1)
        # - Indices 1-3: real_A (preserved RGB for b2, b3, b4)
        # - Indices 4-12: fake_extra[:, 1:, :, :] (G-generated for b5 ... b13)
        fake_B = torch.empty(real_A.size(0), 13, real_A.size(2), real_A.size(3), device=device)
        fake_B[:, 0, :, :] = fake_extra[:, 0, :, :]
        fake_B[:, 1:4, :, :] = real_A
        fake_B[:, 4:, :, :] = fake_extra[:, 1:, :, :]

        loss_GAN_G = criterion_GAN(D_B(fake_B), valid)

        # --- Mapping B -> A ---
        fake_A = F(real_B)  # F maps 13-channel Sentinel-2 to 3-channel RGB
        loss_GAN_F = criterion_GAN(D_A(fake_A), valid)

        # --- Cycle Consistency Loss ---
        recov_A = F(fake_B)  # Cycle A -> B -> A: F(fake_B) should equal real_A
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        
        fake_extra_2 = G(fake_A)  # Cycle B -> A -> B: G(fake_A) generates extra channels from fake_A
        recov_B = torch.empty(real_B.size(0), 13, real_B.size(2), real_B.size(3), device=device)
        recov_B[:, 0, :, :] = fake_extra_2[:, 0, :, :]
        recov_B[:, 1:4, :, :] = fake_A
        recov_B[:, 4:, :, :] = fake_extra_2[:, 1:, :, :]
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) * lambda_cycle

        # Total generator loss
        loss_G = loss_identity + loss_GAN_G + loss_GAN_F + loss_cycle
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Discriminator A Training
        # -----------------------
        optimizer_D_A.zero_grad()
        loss_D_A_real = criterion_GAN(D_A(real_A), valid)
        loss_D_A_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A_total = (loss_D_A_real + loss_D_A_fake) * 0.5
        loss_D_A_total.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Discriminator B Training
        # -----------------------
        optimizer_D_B.zero_grad()
        loss_D_B_real = criterion_GAN(D_B(real_B), valid)
        loss_D_B_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B_total = (loss_D_B_real + loss_D_B_fake) * 0.5
        loss_D_B_total.backward()
        optimizer_D_B.step()

        if i % 50 == 0:
            print(f"[{band_folder}] Epoch [{epoch}/{n_epochs}] Batch [{i}/{len(dataloader)}] Loss_G: {loss_G.item():.4f} Loss_D_A: {loss_D_A_total.item():.4f} Loss_D_B: {loss_D_B_total.item():.4f}")
        
    # -----------------------
    # Save outputs at the end of each epoch
    # -----------------------
    output_dir = f'/content/drive/MyDrive/data_cycle_gan_sampled/output/{os.path.basename(band_folder)}'
    os.makedirs(output_dir, exist_ok=True)

    # Select the first sample from the batch for saving (shape: (13, H, W))
    fake_B_sample = fake_B[0]

    # For true-color visualization, we want to use b2, b3, b4.
    # b2 → index 1, b3 → index 2, b4 → index 3.
    # True-color order: [Red, Green, Blue] = [b4, b3, b2] = indices [3, 2, 1].
    rgb_image = fake_B_sample[[3, 2, 1], :, :]
    # Manually scale from [-1,1] to [0,1]
    rgb_image = (rgb_image + 1) / 2.0
    save_image(rgb_image.cpu(), f'{output_dir}/fakeB_rgb_epoch_{epoch}.png', normalize=False)

    # Save full 13-channel image as TIFF:
    fake_B_np = fake_B_sample.cpu().detach().numpy()  # shape: (13, H, W)
    fake_B_np = np.transpose(fake_B_np, (1, 2, 0))       # shape: (H, W, 13)
    fake_B_np = (fake_B_np + 1) / 2.0  # Scale from [-1,1] to [0,1]
    fake_B_np = fake_B_np.astype(np.float32)
    tiff.imwrite(f'{output_dir}/fakeB_full_epoch_{epoch}.tif', fake_B_np)
    
os.makedirs('models', exist_ok=True)
torch.save(G.state_dict(), f'models/G_{os.path.basename(band_folder)}.pth')
torch.save(F.state_dict(), f'models/F_{os.path.basename(band_folder)}.pth')