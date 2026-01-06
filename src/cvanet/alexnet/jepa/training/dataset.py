import os
import logging
import random
import math
from typing import Tuple, List, Set
from multiprocessing import Value
import numpy as np
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
_GLOBAL_SEED = 0
logger = logging.getLogger(__name__)


class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size
        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    # logger.warning(
                    #     f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batchs: int=1):
        '''
        Create encoder and predictor masks when collating imgs into a batch.
        # 1. sample enc block (size + location) using seed.
        # 2. sample pred block (size) using seed.
        # 3. sample several enc block locations for each image (w/o seed).
        # 4. sample several pred block locations for each image (w/o seed).
        # 5. return enc mask and pred mask.
        '''
        # B = len(batch)
        # collated_batch = torch.utils.data.default_collate(batch)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(generator=g, scale=self.pred_mask_scale, aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(generator=g, scale=self.enc_mask_scale, aspect_ratio_scale=(1., 1.))
        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(batchs):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)
            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        return collated_masks_enc, collated_masks_pred


def apply_mask_to_image(image_tensor, mask_indices, patch_size=16, mask_value=0.3):
    """
    Apply a mask to an image by darkening non-masked regions
    
    Args:
        image_tensor: Tensor of shape (C, H, W)
        mask_indices: 1D tensor containing patch indices to keep visible
        patch_size: Size of each patch
        mask_value: Brightness value for masked-out regions (0-1)
    
    Returns:
        masked_img: Numpy array of masked image (H, W, C)
    """
    # Convert image tensor to numpy
    img = image_tensor.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
    H, W = img.shape[:2]
    h_patches = H // patch_size
    w_patches = W // patch_size
    # Convert mask indices to 2D mask
    mask_flat = mask_indices.numpy()
    mask_2d = np.zeros(h_patches * w_patches)
    mask_2d[mask_flat] = 1
    mask_2d = mask_2d.reshape(h_patches, w_patches)
    # Upsample mask to image size
    mask_upsampled = np.kron(mask_2d, np.ones((patch_size, patch_size)))
    # Apply mask to image
    masked_img = img.copy()
    mask_3d = np.expand_dims(mask_upsampled, axis=2)
    masked_img = masked_img * mask_3d + (1 - mask_3d) * mask_value
    masked_img = torch.tensor(masked_img, dtype=torch.float32).permute(2, 0, 1)
    return masked_img


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
        # masked_image.save('masked.jpg')
        return masked_array


class MultiViewTransform:
    """
    Génère deux vues augmentées de la même image.
    """

    def __init__(self, size=224, patch_size: int=16, mask_value: float=0.3):
        self._patch_size = patch_size
        self._mask_value = mask_value
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)),
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Initialize mask collator
        self.mask_collator = MaskCollator(
            input_size=size,
            patch_size=patch_size,
            enc_mask_scale=(0.2, 0.8),
            pred_mask_scale=(0.2, 0.8),
            aspect_ratio=(0.3, 3.0),
            nenc=1,
            npred=4,
            min_keep=4,
            allow_overlap=False
        )

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.transform(x)
        masks_enc, masks_pred = self.mask_collator()
        y1 = apply_mask_to_image(x, mask_indices=masks_enc[0], patch_size=self._patch_size, mask_value=self._mask_value)
        y2 = apply_mask_to_image(x, mask_indices=masks_pred[0], patch_size=self._patch_size, mask_value=self._mask_value)
        # save_image(y1, 'y1.png')
        # save_image(y2, 'y2.png')
        return y1, y2


class CustomImageDataset(BaseDataset):
    """
    Dataset personnalisé pour les images avec transformations multi-vues.
    """

    def __init__(self, image_files: List[str], size=224):
        self._image_files = image_files
        self._transform = MultiViewTransform(size)

    def __len__(self) -> int:
        # return 1000
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
