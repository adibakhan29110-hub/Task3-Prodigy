import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.utils import save_image

from options import get_options
from dataset import PairedImageDataset
from models import GeneratorUNet, Discriminator
from losses import GANLoss

# -------------------
# Utilities
# -------------------
def save_checkpoint(model, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    print(f"Saved checkpoint: {path}")


def save_samples(gen, inputs, targets, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        gen.eval()
        outputs = gen(inputs)
        combined = torch.cat([inputs, targets, outputs], 0)
        save_image(combined, os.path.join(output_dir, f"epoch_{epoch}.png"), nrow=inputs.size(0), normalize=True)
    gen.train()


# -------------------
# Training
# -------------------
def train():
    opt = get_options()
    device = torch.device(opt.device)

    # Dataset
    dataset = PairedImageDataset(opt.dataroot, img_size=opt.img_size)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # Models
    gen = GeneratorUNet().to(device)
    disc = Discriminator().to(device)

    # Optimizers
    g_optimizer = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    d_optimizer = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # Losses
    gan_loss = GANLoss().to(device)
    l1_loss = nn.L1Loss()

    print("Starting training...")
    for epoch in range(1, opt.epochs + 1):
        for i, (inp, tar) in enumerate(loader):
            inp, tar = inp.to(device), tar.to(device)

            # -----------------
            # Train Discriminator
            # -----------------
            d_optimizer.zero_grad()
            fake = gen(inp)
            real_loss = gan_loss(disc(inp, tar), True)
            fake_loss = gan_loss(disc(inp, fake.detach()), False)
            d_loss = (real_loss + fake_loss) * 0.5
            d_loss.backward()
            d_optimizer.step()

            # -----------------
            # Train Generator
            # -----------------
            g_optimizer.zero_grad()
            g_loss = gan_loss(disc(inp, fake), True) + opt.lambda_l1 * l1_loss(fake, tar)
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch}/{opt.epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # Save checkpoint and sample every 10 epochs
        if epoch % opt.save_epoch_freq == 0 or epoch == opt.epochs:
            save_checkpoint(gen, g_optimizer, os.path.join(opt.checkpoint_dir, "latest.pth"))
            save_samples(gen, inp, tar, epoch, opt.output_dir + "/samples")


if __name__ == "__main__":
    train()


