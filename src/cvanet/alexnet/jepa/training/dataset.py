import os
import random
from typing import Tuple, List, Set
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as BaseDataset, DataLoader
from torchvision import transforms

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp',
    '.tiff', '.tif', '.webp', '.svg', '.ico',
    '.heic', '.heif', '.raw', '.cr2', '.nef',
    '.arw', '.psd', '.ai', '.eps'
}

class ImageRegionMasker(torch.nn.Module):
    """
    Transform images by randomly masking several large regions with black pixels,
    leaving only one region intact.
    """

    def __init__(self, num_regions_to_show=4, num_masked_regions=12, min_region_size=0.2, max_region_size=0.4):
        """
        Initialize the masker with configuration parameters.

        Args:
            num_regions_to_show: Number of regions to keep visible/intact (default: 1)
            num_masked_regions: Number of regions to mask with black (default: 8)
            min_region_size: Minimum region size as fraction of image dimensions (default: 0.2)
            max_region_size: Maximum region size as fraction of image dimensions (default: 0.5)
        """
        super().__init__()
        self.num_regions_to_show = num_regions_to_show
        self.num_masked_regions = num_masked_regions
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size

    def _generate_random_region(self, h, w):
        """Generate random rectangular region coordinates."""
        region_h = random.randint(int(h * self.min_region_size), int(h * self.max_region_size))
        region_w = random.randint(int(w * self.min_region_size), int(w * self.max_region_size))

        y1 = random.randint(0, h - region_h)
        x1 = random.randint(0, w - region_w)
        y2 = y1 + region_h
        x2 = x1 + region_w

        return (y1, y2, x1, x2)

    def _regions_overlap(self, region1, region2):
        """Check if two regions overlap."""
        y1_1, y2_1, x1_1, x2_1 = region1
        y1_2, y2_2, x1_2, x2_2 = region2

        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

    def forward(self, img):
        """
        Apply transformation: mask multiple regions, keep specified number intact.
        """
        # Load image using PIL
        # img = Image.open(image_path)
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        # Generate all regions (masked + intact)
        total_regions = self.num_masked_regions + self.num_regions_to_show
        regions = []
        max_attempts = 100
        for i in range(total_regions):
            attempts = 0
            while attempts < max_attempts:
                region = self._generate_random_region(h, w)
                # Check for overlaps with existing regions;
                overlap = False
                for existing_region in regions:
                    if self._regions_overlap(region, existing_region):
                        overlap = True
                        break
                if not overlap:
                    regions.append(region)
                    break
                attempts += 1
            if attempts == max_attempts and len(regions) < i + 1:
                # If we can't find non-overlapping region, allow overlap;
                regions.append(self._generate_random_region(h, w))
        # Randomly choose which regions stay intact
        intact_indices = random.sample(range(len(regions)), self.num_regions_to_show)
        intact_regions = [regions[idx] for idx in intact_indices]
        # Create mask: start with all black;
        masked_array = np.zeros_like(img_array)
        # Copy the intact regions from original image;
        for intact_region in intact_regions:
            y1, y2, x1, x2 = intact_region
            masked_array[y1:y2, x1:x2] = img_array[y1:y2, x1:x2]
        masked_image = Image.fromarray(masked_array.astype(np.uint8))
        # masked_image.save('masked.jpg')
        return masked_image


class MultiViewTransform:
    """
    Génère deux vues augmentées de la même image.
    """

    def __init__(self, size=224):
        self.context_transform = transforms.Compose([
            ImageRegionMasker(),
            # transforms.Resize((size, size)),
            # transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = transforms.Compose([
            # transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._size = (size, size)

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.resize(self._size)
        return self.context_transform(x), self.target_transform(x)


class CustomImageDataset(BaseDataset):
    """
    Dataset personnalisé pour les images avec transformations multi-vues.
    """

    def __init__(self, image_files: List[str], size=224):
        self._image_files = image_files
        self._transform = MultiViewTransform(size)

    def __len__(self) -> int:
        # return 100
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self._image_files[idx]
        image = Image.open(image_file).convert('RGB')
        images = self._transform(image)
        return images


def enumerate_image_files(directory_path: str, recursive: bool = True,extensions: Set[str] = None) -> List[str]:
    """
    Enumerate all image files in a directory and its subdirectories.

    This function uses os.walk() to recursively traverse the directory structure and collects files
    with image extensions.

    Args:
        directory_path: Path to the directory to search.
        recursive: If True, search in subdirectories (default: True).
        extensions: Set of image file extensions to look for. If None, uses default image extensions.

    Returns:
        List of full paths to image files found.

    Raises:
        FileNotFoundError: If directory_path doesn't exist.
        NotADirectoryError: If directory_path is not a directory.

    Example:
        >>> files = enumerate_image_files("/path/to/photos")
        >>> print(f"Found {len(files)} image files")
    """
    # Validate directory exists and is a directory;
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")
    # Default image extensions if none provided;
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    image_files = []
    # Use os.walk to traverse directory tree;
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            # Get file extension and convert to lowercase for case-insensitive comparison;
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in extensions:
                # Construct full path and add to list;
                full_path = os.path.join(dirpath, filename)
                image_files.append(full_path)
        # If not recursive, break after first level;
        if not recursive:
            # Clear dirnames to prevent os.walk from going deeper;
            dirnames.clear()
    return image_files


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
    image_files = enumerate_image_files(train_data_dir)
    train_dataset = CustomImageDataset(image_files.copy(), size=img_size)
    image_files = enumerate_image_files(val_data_dir)
    val_dataset = CustomImageDataset(image_files.copy(), size=img_size)
    train_dataset_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dataset_loader, val_dataset_loader
