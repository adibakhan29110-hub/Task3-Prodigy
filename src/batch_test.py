# src/batch_test.py
import os
import torch
from PIL import Image
from torchvision import transforms
from models import GeneratorUNet

# -------------------------
# Paths
# -------------------------
checkpoint = "checkpoints/latest.pth"
input_dir = "data/sample_pairs"
output_dir = "outputs/batch_results"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load generator
# -------------------------
gen = GeneratorUNet().to(device)
checkpoint_data = torch.load(checkpoint, map_location=device)
gen.load_state_dict(checkpoint_data['model_state_dict'])
gen.eval()

# -------------------------
# Image transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# -------------------------
# Process all images
# -------------------------
for file in os.listdir(input_dir):
    if file.lower().endswith(('jpg','png','jpeg')):
        img_path = os.path.join(input_dir, file)
        img = Image.open(img_path).convert('RGB')

        # Crop left half if side-by-side
        w, h = img.size
        if w > h:
            w2 = int(w/2)
            inp = img.crop((0,0,w2,h))
        else:
            inp = img

        inp_tensor = transform(inp).unsqueeze(0).to(device)

        with torch.no_grad():
            out_tensor = gen(inp_tensor)
        out_tensor = (out_tensor + 1)/2.0  # rescale to [0,1]

        save_path = os.path.join(output_dir, file)
        transforms.ToPILImage()(out_tensor.squeeze(0).cpu()).save(save_path)
        print(f"Saved: {save_path}")

print("\n✅ Batch testing complete. Check outputs/batch_results/")
