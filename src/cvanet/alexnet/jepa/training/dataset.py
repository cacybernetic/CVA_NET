import os
from typing import Tuple
from torch.utils.data import Dataset as BaseDataset, DataLoader
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
        return 1000
        return len(self.image_folder)

    def __getitem__(self, idx):
        img, _ = self.image_folder[idx]
        if self.transform:
            imgs = self.transform(img)
        return imgs


def custom_dataloaders(
    train_data_dir: str,
    val_data_dir: str,
    img_size: int=224,
    batch_size: int=32,
    num_workers: int=2,
    pin_memory: bool=False,
) -> Tuple[DataLoader, DataLoader]:
    assert train_data_dir, "No training dataset directory provided."
    assert val_data_dir, "No validation dataset directory provided."
    if not os.path.isdir(train_data_dir):
        raise FileNotFoundError("No such training dataset directory at \"%s\"." % (train_data_dir,))
    if not os.path.isdir(val_data_dir):
        raise FileNotFoundError("No such validation dataset directory at \"%s\"." % (val_data_dir,))
    train_dataset = CustomImageDataset(
        root_dir=train_data_dir, transform=MultiViewTransform(size=img_size))
    val_dataset = CustomImageDataset(
        root_dir=val_data_dir, transform=MultiViewTransform(size=img_size))
    train_dataset_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dataset_loader, val_dataset_loader
