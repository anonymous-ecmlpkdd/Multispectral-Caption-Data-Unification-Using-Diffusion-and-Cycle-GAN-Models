import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Görüntü (.jpg) ve caption (.txt) dosyalarının bulunduğu klasör.
        transform: Görüntüye uygulanacak dönüşümler (örneğin, yeniden boyutlandırma, normalize).
        """
        self.root_dir = root_dir
        self.transform = transform
        # Sadece .jpg uzantılı dosyaları alıyoruz
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        caption_filename = image_filename.replace('.jpg', '.txt')
        
        image_path = os.path.join(self.root_dir, image_filename)
        caption_path = os.path.join(self.root_dir, caption_filename)
        
        # Görüntüyü açıp RGB formatına çeviriyoruz
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Caption dosyasını okuyoruz
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        
        return {"image": image, "caption": caption}

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # Görüntüleri [-1, 1] aralığına normalize ediyoruz
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Stable Diffusion 3.5 modelini kullanmak için model id'nizi girin.
model_id = "stabilityai/stable-diffusion-2.1-base"  # Uygun model id ile değiştirin.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer ve Text Encoder: Caption'ları işlemek için.
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

# VAE (latent uzay dönüşümü) ve UNet (gürültü tahmini) modellerini yüklüyoruz.
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)

# Diffusion sürecinde kullanılacak noise scheduler'ı yüklüyoruz.
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# VAE ve text_encoder’ın ağırlıklarını donduruyoruz (fine-tuning sırasında güncellenmeyecekler).
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Veri setinizin bulunduğu klasörün yolunu belirleyin.
data_root = "path/to/your/dataset"  # Bu yolu kendi verisetinizin yolu ile değiştirin.
dataset = SatelliteDataset(root_dir=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

accelerator = Accelerator()
optimizer = optim.AdamW(unet.parameters(), lr=1e-5)
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

num_epochs = 1  # Deneme için 1 epoch, tam eğitimde artırabilirsiniz.
global_step = 0

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # Görüntü ve caption çiftlerini alıyoruz
        images = batch["image"].to(device)
        captions = batch["caption"]

        # Caption'ları tokenize ediyoruz
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        # Text Encoder ile caption'ların embedding'lerini elde ediyoruz
        with torch.no_grad():
            encoder_hidden_states = text_encoder(text_input_ids)[0]

        # Görüntüleri VAE ile latent uzaya encode ediyoruz
        latents = vae.encode(images).latent_dist.sample()
        # Stable Diffusion'da kullanılan ölçek faktörü (320x320 için ince ayar gerekebilir)
        latents = latents * 0.18215

        # Aynı boyutta rastgele gürültü oluşturuyoruz
        noise = torch.randn(latents.shape).to(device)
        # Her görüntü için rastgele timestep seçiyoruz
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()

        # Noise scheduler ile latent'lere gürültü ekliyoruz (forward diffusion)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # UNet modelinden, gürültülü latent ve text embedding'lere göre gürültü tahmini alıyoruz
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Tahmin edilen gürültü ile gerçek gürültü arasındaki farkı MSE loss ile hesaplıyoruz
        loss = nn.MSELoss()(noise_pred, noise)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        global_step += 1

        if global_step % 100 == 0:
            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item()}")

output_dir = "./fine_tuned_unet"
os.makedirs(output_dir, exist_ok=True)
accelerator.wait_for_everyone()  # Dağıtık eğitimde senkronizasyon
unet.save_pretrained(output_dir)

print("Fine-tuning tamamlandı, model kaydedildi.")

