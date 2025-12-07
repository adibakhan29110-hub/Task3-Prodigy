import os
import torch
from torchvision.utils import save_image


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, path)
    print(f"Checkpoint saved: {path}")


def save_sample_images(inp, fake, target, out_dir, epoch):
    os.makedirs(out_dir, exist_ok=True)

    # Un-normalize
    inp = inp * 0.5 + 0.5
    fake = fake * 0.5 + 0.5
    target = target * 0.5 + 0.5

    save_image(inp, f"{out_dir}/epoch_{epoch}_input.png")
    save_image(fake, f"{out_dir}/epoch_{epoch}_fake.png")
    save_image(target, f"{out_dir}/epoch_{epoch}_target.png")

    print(f"Saved sample images for epoch {epoch}")
