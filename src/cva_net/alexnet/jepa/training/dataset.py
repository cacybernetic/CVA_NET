# import os
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class MultiViewTransform:
    """
    Génère deux vues augmentées de la même image.
    """

    def __init__(self, size=224):
        self.context_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.context_transform(x), self.target_transform(x)


class CustomImageDataset(BaseDataset):
    """
    Dataset personnalisé pour les images avec transformations multi-vues.
    """

    def __init__(self, root_dir, transform=None):
        self.image_folder = ImageFolder(root_dir)
        self.transform = transform
        # self.class_names = os.listdir(root_dir)
        self.class_names = self.image_folder.classes

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        img, label = self.image_folder[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
