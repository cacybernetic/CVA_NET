import torch
import math
import logging
from multiprocessing import Value
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
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
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        return mask, mask_complement

    def __call__(self, batch):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(generator=g, scale=self.pred_mask_scale, 
                                         aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(generator=g, scale=self.enc_mask_scale, 
                                         aspect_ratio_scale=(1., 1.))
        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
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
                    acceptable_regions = None
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
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        return collated_batch, collated_masks_enc, collated_masks_pred


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
    
    return masked_img


def render_masks(image_tensor, masks_enc, masks_pred, patch_size=16, save_path='mask_visualization.png'):
    """
    Render and display encoder and predictor masks on the original image
    
    Args:
        image_tensor: Tensor of shape (C, H, W)
        masks_enc: Encoder masks, shape (nenc, num_patches)
        masks_pred: Predictor masks, shape (npred, num_patches)
        patch_size: Size of each patch
        save_path: Path to save the visualization
    """
    # Get original image
    img = image_tensor.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    
    nenc = masks_enc.shape[0]
    npred = masks_pred.shape[0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, max(nenc, npred) + 1, figsize=(15, 8))
    if nenc == 1 and npred == 1:
        axes = axes.reshape(2, -1)
    
    # Show original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    # Apply and visualize encoder masks
    for i in range(nenc):
        masked_img = apply_mask_to_image(image_tensor, masks_enc[i], patch_size)
        axes[0, i+1].imshow(masked_img)
        axes[0, i+1].set_title(f'Encoder Mask {i+1}')
        axes[0, i+1].axis('off')
    
    # Apply and visualize predictor masks
    for i in range(npred):
        masked_img = apply_mask_to_image(image_tensor, masks_pred[i], patch_size)
        axes[1, i+1].imshow(masked_img)
        axes[1, i+1].set_title(f'Predictor Mask {i+1}')
        axes[1, i+1].axis('off')
    
    # Hide extra subplots
    for i in range(max(nenc, npred) + 1):
        if i > nenc:
            axes[0, i].axis('off')
        if i > npred:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as '{save_path}'")
    plt.show()


def main():
    # Configuration
    input_size = (224, 224)
    patch_size = 16
    
    # Initialize mask collator
    mask_collator = MaskCollator(
        input_size=input_size,
        patch_size=patch_size,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False
    )
    
    # Load and preprocess an image
    # You can replace this with your own image path
    # For demo, create a synthetic image
    # print("Creating synthetic test image...")
    # img = Image.new('RGB', input_size, color='white')
    # pixels = img.load()
    # for i in range(input_size[0]):
    #     for j in range(input_size[1]):
    #         pixels[j, i] = (
    #             int(255 * i / input_size[0]),
    #             int(255 * j / input_size[1]),
    #             128
    #         )
    
    # Alternatively, load a real image:
    img = Image.open('image.jpg').resize(input_size)
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)
    
    # Create a batch with single image
    batch = [img_tensor]
    
    # Generate masks
    print("Generating masks...")
    collated_batch, masks_enc, masks_pred = mask_collator(batch)
    
    # print(f"Batch shape: {collated_batch.shape}")
    # print(f"Encoder masks shape: {masks_enc.shape}")
    # print(f"Predictor masks shape: {masks_pred.shape}")
    
    # Visualize results
    print("Creating visualization...")
    render_masks(
        img_tensor,
        masks_enc[0],  # First image in batch
        masks_pred[0],  # First image in batch
        patch_size=patch_size
    )
    
    # Example: Apply a single mask to the image
    print("\nExample: Applying first encoder mask...")
    masked_img = apply_mask_to_image(img_tensor, masks_enc[0, 0], patch_size)
    
    # You can save or further process the masked image
    plt.figure(figsize=(6, 6))
    plt.imshow(masked_img)
    plt.title('Single Encoder Mask Applied')
    plt.axis('off')
    plt.savefig('single_mask_example.png', dpi=150, bbox_inches='tight')
    print("Single mask example saved as 'single_mask_example.png'")
    plt.show()


if __name__ == "__main__":
    main()
