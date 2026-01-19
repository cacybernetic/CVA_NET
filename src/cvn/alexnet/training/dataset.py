import os
import logging
import random
from typing import Tuple, List, Set, Dict, Callable, Any
from PIL import Image
import torch
from torch.utils.data import Dataset as BaseDataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp',
    '.tiff', '.tif', '.webp', '.svg', '.ico',
    '.heic', '.heif', '.raw', '.cr2', '.nef',
    '.arw', '.psd', '.ai', '.eps'
}

LOGGER = logging.getLogger(__name__)


class ImageTransformation:
    """
    Génère deux vues augmentées de la même image.
    """

    def __init__(self, size=224, image_mode: str="RGB", train: bool=True) -> None:
        self._pipeline_transform = None
        if image_mode.upper() == 'RGB':
            if train:
                self._pipeline_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((size + 100, size + 100)),
                    transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    # transforms.RandomGrayscale(p=0.2),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self._pipeline_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((size, size)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        elif image_mode.upper() == 'L':
            if train:
                self._pipeline_transform = transforms.Compose([
                    # Convertir en tensor (shape: [1, 224, 224]);
                    transforms.ToTensor(),
                    # Redimensionner l'image à une taille légèrement supérieure;
                    transforms.Resize((size + 100, size + 100)),
                    # Data Augmentation pour améliorer la généralisation;
                    transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    # Augmentation de contraste et luminosité (important pour grayscale);
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    # Normalisation pour images grayscale (1 canal);
                    # Utiliser mean et std pour un seul canal;
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            else:
                self._pipeline_transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.CenterCrop(224),
                    transforms.Resize((size, size)),
                    # Normalisation identique;
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
        else:
            raise NotImplementedError("The pipeline of transformation for this image mode is not implemented yet.")

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._pipeline_transform(x)
        return x

class Dataset(BaseDataset):
    """
    Dataset personnalisé pour les images avec transformations multi-vues.
    """

    def __init__(
        self,
        image_files: List[str],
        class_ids: List[int],
        img_channels: int=3,
        transform: Callable[[Image.Image], torch.Tensor]=None
    ) -> None:
        self._image_files = image_files
        self._class_ids = class_ids
        self._img_channels = img_channels
        self._transform = transform

    def __len__(self) -> int:
        # return 1000
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self._image_files[idx]
        class_id = self._class_ids[idx]
        image = Image.open(image_file)
        if self._img_channels == 3:
            image = image.convert('RGB')
        elif self._img_channels == 1:
            image = image.convert('L')
        image = self._transform(image)
        class_id = torch.tensor(class_id, dtype=torch.int64)
        return image, class_id


def _check_image_file(file_name: str) -> bool:
    file_name = file_name.split('.')
    extension = '.' + file_name[-1]
    return extension in IMAGE_EXTENSIONS


def _load_labeled_image_files(directory_path: str, extensions: Set[str]=None) -> Dict[str, list]:
    """
    Enumerate all image files in a directory and its subdirectories.

    This function uses os.walk() to recursively traverse the directory structure and collects files
    with image extensions.

    Args:
        directory_path: Path to the directory to search.
        recursive: If True, search in subdirectories (default: True).
        extensions: Set of image file extensions to look for. If None, uses default image extensions.

    Returns:
        image_files: List of full paths to image files found;
        class_ids: List of the class ids for each image;
        class_names: list of the class names found.

    Raises:
        FileNotFoundError: If directory_path doesn't exist.
        NotADirectoryError: If directory_path is not a directory.

    Example:
        >>> files = enumerate_image_files("/path/to/photos")
        >>> print(f"Found {len(files)} image files")
    """
    # Validate directory exists and is a directory;
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory does not exist: {directory_path}.")
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Path is not a directory: {directory_path}.")
    # Default image extensions if none provided;
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    class_names = sorted(os.listdir(directory_path))
    image_files = []
    class_ids = []
    for class_dn in class_names:
        class_dir = os.path.join(directory_path, class_dn)
        file_names = os.listdir(class_dir)
        file_paths = [os.path.join(class_dir, fn) for fn in file_names if _check_image_file(fn)]
        image_files.extend(file_paths)
        class_ids.extend([class_names.index(class_dn)] * len(file_paths))
    return {
        'image_files': image_files,
        'class_ids': class_ids,
        'class_names': class_names}


def _get_validation_samples(image_files: List[str], class_ids: List[int], p: float) -> Tuple[List[str], List[int]]:
    ds_dataset_len = len(image_files)
    val_dataset_len = int(p * ds_dataset_len)
    indices = torch.randint(0, val_dataset_len, (val_dataset_len,))
    # indices = torch.arange(0, val_dataset_len, 1)
    val_image_files = []
    val_class_ids = []
    for index in indices:
        val_image_files.append(image_files[index.item()])
        val_class_ids.append(class_ids[index.item()])
    return val_image_files, val_class_ids


def build(
    train_data_dir: str,
    test_data_dir: str,
    img_channels: int=3,
    img_size: int=224,
    batch_size: int=32,
    val: float=0.1,
    num_workers: int=2,
    pin_memory: bool=False,
    apply_transformations: bool=True,
) -> Dict[str, Any]:
    assert train_data_dir, "No training dataset directory provided."
    assert test_data_dir, "No validation dataset directory provided."
    if not os.path.isdir(train_data_dir):
        raise FileNotFoundError("No such training dataset directory at \"%s\"." % (train_data_dir,))
    if not os.path.isdir(test_data_dir):
        raise FileNotFoundError("No such evalutation dataset directory at \"%s\"." % (test_data_dir,))
    train_transform = None
    test_transform = None
    if apply_transformations:
        mode = 'RGB' if img_channels == 3 else 'L'
        train_transform = ImageTransformation(img_size, mode, train=True)
        test_transform = ImageTransformation(img_size, mode)
    ## Build training set;
    train_samples = _load_labeled_image_files(train_data_dir)
    train_dataset = Dataset(
        train_samples['image_files'], train_samples['class_ids'], img_channels=img_channels, transform=train_transform)
    ## Build test set;
    test_samples = _load_labeled_image_files(test_data_dir)
    test_dataset = Dataset(
        test_samples['image_files'], test_samples['class_ids'], img_channels=img_channels, transform=test_transform)
    ## Build validation set;
    val_images, val_class_ids = _get_validation_samples(test_samples['image_files'], test_samples['class_ids'], p=val)
    val_dataset = Dataset(
        val_images, val_class_ids, img_channels=img_channels, transform=test_transform)
    ## Build data loaders;
    train_dataset_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dataset_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return {
        'train_data_loader': train_dataset_loader,
        'val_data_loader': val_dataset_loader,
        'test_data_loader': test_dataset_loader,
        'class_names': train_samples['class_names']}
