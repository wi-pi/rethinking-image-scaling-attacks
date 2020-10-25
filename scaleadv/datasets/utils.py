from torchvision.datasets import ImageFolder


class ImageFolderWithIndex(ImageFolder):

    def __getitem__(self, index):
        item = super(ImageFolderWithIndex, self).__getitem__(index)
        return (index,) + item
