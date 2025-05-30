# Multispectral-Caption-Data-Unification-Using-Diffusion-and-Cycle-GAN-Models

## 📜 Overview

**Multispectral Caption-Image Unification via Diffusion and CycleGAN** proposes a full multimodal pipeline that enables the generation and unification of satellite image data across three modalities:
- **Caption (Text)**
- **RGB Image**
- **Multispectral Sentinel-2 Image**

The system integrates **fine-tuned Stable Diffusion** for text-to-RGB image generation and **CycleGAN** for RGB-to-multispectral translation.  
It allows **triplet data creation** even when only partial information (e.g., just caption or RGB) is available.

---

## 🔗 Resources

- 📄 [Full Paper (Preprint)](link_to_paper_if_available)
- 🧠 [Training Code (GitHub)](link_to_paper_if_available)

---

## 📜 Citation

If you use this model, please cite:

```bibtex
@misc{will be added
}
```

## 🚀 Key Features

- **Caption ➔ RGB Image ➔ Multispectral Image** generation
- **RGB Image ➔ Caption** and **Multispectral Image** generation
- **Multispectral Image ➔ RGB Image ➔ Caption** reconstruction
- Fine-tuned **Stable Diffusion 2-1 Base** on satellite captions
- Custom **CycleGAN** model trained for Sentinel-2 13-band spectral transformation
- Specialized **SAM Loss** (Spectral Angle Mapper) for better multispectral consistency
- Supports creating fully unified datasets from previously disconnected modalities

---

## 📚 Training Details

- **Stable Diffusion Fine-Tuning:**  
  - Dataset: 675,000 SkyScript images with captions generated by **Qwen2-VL-2B-Instruct**
  - Training: Text-to-Image generation targeting satellite domain
  
- **CycleGAN Training:**  
  - Dataset: 120,000 generated RGB images + 27,000 Eurosat multispectral images
  - Special Loss: Spectral Angle Mapper (SAM) loss 
  - Resolution: 64×64 crops during training, 512×512 sliding window inference
  
- **Hardware:**  
  - Google Colab Pro+  
  - NVIDIA A100 GPU  

---

## 🛰️ Applications

- Synthetic satellite dataset generation
- Remote sensing research (land cover classification, environmental monitoring)
- Data augmentation for multispectral models
- Disaster monitoring and environmental change detection

---

## 🧩 Model Components

| Component | Description |
|:---|:---|
| `stable-diffusion-finetuned-satellite` | Fine-tuned Stable Diffusion 2-1 Base model for satellite image synthesis |
| `cyclegan-rgb-to-multispectral` | Custom CycleGAN for RGB to multispectral (Sentinel-2) translation |
| `synthetic-triplet-dataset` | 120,000 RGB + multispectral + caption synthetic triplet dataset |

---

## ⚡ Quick Example: Generate an Image from a Single Caption

```python
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file as safe_load
from PIL import Image
import os

# Load fine-tuned UNet weights
checkpoint_path = "path/to/your/model.safetensors"

# Initialize UNet and load weights
base_unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="unet",
    torch_dtype=torch.float16
)
state_dict = safe_load(checkpoint_path)
base_unet.load_state_dict(state_dict)
unet = base_unet

# Load VAE, text encoder, tokenizer, scheduler
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="vae",
    torch_dtype=torch.float16
)
text_encoder = CLIPTextModel.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="text_encoder",
    torch_dtype=torch.float16
)
tokenizer = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="tokenizer"
)
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    subfolder="scheduler"
)

# Build the pipeline
pipe = StableDiffusionPipeline(
    unet=unet,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None
).to("cuda")

# Single caption prompt
prompt = "A coastal city with large harbors and residential areas visible from space"

# Generate the image
result = pipe(prompt, num_inference_steps=50)
image = result.images[0]

# Save the image
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "sample_generated_image.png")
image.save(output_path)

print(f"✅ Image generated and saved at {output_path}")

```

## ⚡ Quick Example: RGB-to-Multispectral Conversion with CycleGAN

```python
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from safetensors.torch import safe_open  # for loading .safetensors weights

# ---------------------------
# Model & Input Settings
# ---------------------------
model_path = "path/to/G_eurosat_ms.safetensors"  # update to your model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your Generator (3→13 channels)
G = Generator(input_nc=3, output_nc=10).to(device)
with safe_open(model_path, framework="pt", device="cpu") as f:
    state_dict = {k: f.get_tensor(k) for k in f.keys()}
G.load_state_dict(state_dict)
G.eval()

# Load an RGB test image
rgb_path = "path/to/sample_rgb.jpg"
input_image = Image.open(rgb_path).convert("RGB").resize((512, 512))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
input_tensor = transform(input_image).unsqueeze(0).to(device)  # (1,3,512,512)

# ---------------------------
# Sliding-Window Inference
# ---------------------------
patch_size = 64
h, w = 512, 512
output_fake = torch.zeros((13, h, w), device=device)

for y in range(0, h, patch_size):
    for x in range(0, w, patch_size):
        patch = input_tensor[:, :, y:y+patch_size, x:x+patch_size]
        with torch.no_grad():
            extra = G(patch)  # (1,10,64,64)
        # assemble 13-channel patch
        combined = torch.empty(1, 13, patch_size, patch_size, device=device)
        combined[:, 0, :, :] = extra[:, 0, :, :]           # band 1
        combined[:, 1:4, :, :] = patch                    # bands 2–4 (RGB)
        combined[:, 4:, :, :] = extra[:, 1:, :, :]        # bands 5–13
        output_fake[:, y:y+patch_size, x:x+patch_size] = combined.squeeze(0)

# to CPU & normalize from [-1,1] to [0,1]
fake_np = output_fake.cpu().numpy()
fake_np = (fake_np + 1) / 2.0       # shape (13,512,512)
fake_np = np.transpose(fake_np, (1,2,0))  # (512,512,13)

# Optional: save as GeoTIFF
# import tifffile as tiff
# tiff.imwrite("generated_multispectral.tif", fake_np.astype(np.float32))

# ---------------------------
# Spectral Visualization
# ---------------------------
spectral_composites = {
    "Natural Color (B4,B3,B2)": [1,2,3],
    "Color Infrared (B8,B4,B3)": [7,3,2],
    "Short-Wave Infrared (B12,B8A,B4)": [12,8,3],
    "Agriculture (B11,B8,B2)": [10,7,1],
    "Geology (B12,B11,B2)": [12,10,1],
    "Bathymetric (B4,B3,B1)": [3,2,0]
}

# Compute NDVI
ndvi = (fake_np[:,:,7] - fake_np[:,:,3]) / (fake_np[:,:,7] + fake_np[:,:,3] + 1e-6)

fig, axs = plt.subplots(2, 4, figsize=(16,8))
axs = axs.flatten()

# plot each composite
for idx, (title, bands) in enumerate(spectral_composites.items()):
    img = fake_np[:,:,bands] if title.endswith("(B4,B3,B2)") else np.mean(fake_np[:,:,bands], axis=2)
    axs[idx].imshow(img, cmap=None if title.endswith("(B4,B3,B2)") else "inferno")
    axs[idx].set_title(title)
    axs[idx].axis("off")

# plot NDVI
axs[-1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
axs[-1].set_title("Vegetation Index (NDVI)")
axs[-1].axis("off")

plt.tight_layout()
plt.show()
```
