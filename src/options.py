import argparse
import os

def get_options():
    parser = argparse.ArgumentParser(description='pix2pix - cGAN Image-to-Image')

    # Get project root
    root_dir = os.getcwd()

    # Data
    parser.add_argument('--dataroot', type=str, default=os.path.join(root_dir, 'data', 'sample_pairs'), help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--lambda_l1', type=float, default=100.0)

    # Checkpoints / output
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(root_dir, 'checkpoints'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(root_dir, 'outputs'))
    parser.add_argument('--save_epoch_freq', type=int, default=10)

    # Misc
    parser.add_argument('--device', type=str, default='cuda' if __import__('torch').cuda.is_available() else 'cpu')

    return parser.parse_args()
