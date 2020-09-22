import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class ImageFolderWithIndex(ImageFolder):

    def __getitem__(self, index):
        item = super(ImageFolderWithIndex, self).__getitem__(index)
        return (index,) + item


class ImageFilesDataset(Dataset):

    def __init__(self, root, suffix, transform=None):
        total = os.listdir(root)
        self.root = root
        self.imgs = [f for f in total if f.endswith(suffix)]
        self.T = transform if transform else lambda x: x

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        name = self.imgs[item]
        pth = os.path.join(self.root, name)
        return name, self.T(Image.open(pth))


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    ds = ImageFilesDataset(root='static/results/imagenet-600', suffix='.src_inp.png', transform=T.ToTensor())
    ld = DataLoader(ds, batch_size=2)
