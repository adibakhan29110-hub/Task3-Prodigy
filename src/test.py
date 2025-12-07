import torch
from PIL import Image
from torchvision import transforms
from models import GeneratorUNet
import argparse
import os

# -------------------
# Test a single image
# -------------------
def test_single(input_path, output_path, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    gen = GeneratorUNet().to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    gen.load_state_dict(checkpoint_data['model_state_dict'])
    gen.eval()

    # Load input image
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    if w > h:  # side-by-side
        w2 = int(w / 2)
        inp = img.crop((0, 0, w2, h))
    else:
        inp = img

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    inp_tensor = transform(inp).unsqueeze(0).to(device)

    # Generate output
    with torch.no_grad():
        out_tensor = gen(inp_tensor)
    out_tensor = (out_tensor + 1) / 2.0  # rescale to [0,1]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transforms.ToPILImage()(out_tensor.squeeze(0).cpu()).save(output_path)
    print(f"Saved result: {output_path}")


# -------------------
# CLI
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    test_single(args.input, args.output, args.checkpoint)
