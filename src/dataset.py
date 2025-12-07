import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """
    Supports two formats:
    - side-by-side paired images (left=input, right=target)
    - folder-structured pairs: dataroot/input/ and dataroot/target/ with matching filenames
    """
    def __init__(self, root, img_size=256, mode='side_by_side'):
        self.root = root
        self.img_size = img_size
        self.mode = mode

        if mode == 'side_by_side':
            self.files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('jpg','png','jpeg'))]
        else:
            inp_dir = os.path.join(root, 'input')
            tar_dir = os.path.join(root, 'target')
            self.files = [(os.path.join(inp_dir, f), os.path.join(tar_dir, f)) for f in os.listdir(inp_dir)]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.mode == 'side_by_side':
            path = self.files[idx]
            img = Image.open(path).convert('RGB')
            w, h = img.size
            w2 = int(w / 2)
            inp = img.crop((0, 0, w2, h))
            tar = img.crop((w2, 0, w, h))
        else:
            inp_path, tar_path = self.files[idx]
            inp = Image.open(inp_path).convert('RGB')
            tar = Image.open(tar_path).convert('RGB')

        inp = self.transform(inp)
        tar = self.transform(tar)
        return inp, tar