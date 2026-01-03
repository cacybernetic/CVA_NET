import numpy as np
from PIL import Image
import random


class ImageRegionMasker:
    """
    Transform images by randomly masking several large regions with black pixels,
    leaving only one region intact.
    """

    def __init__(self, num_regions_to_show=1, num_masked_regions=8, min_region_size=0.2, max_region_size=0.5):
        """
        Initialize the masker with configuration parameters.

        Args:
            num_regions_to_show: Number of regions to keep visible/intact (default: 1)
            num_masked_regions: Number of regions to mask with black (default: 8)
            min_region_size: Minimum region size as fraction of image dimensions (default: 0.2)
            max_region_size: Maximum region size as fraction of image dimensions (default: 0.5)
        """
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

    def transform(self, image_path, output_path=None):
        """
        Apply transformation: mask multiple regions, keep specified number intact.

        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)

        Returns:
            tuple: (transformed_image_array, list_of_intact_region_coords)
        """
        # Load image using PIL
        img = Image.open(image_path)
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

                # Check for overlaps with existing regions
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
                # If we can't find non-overlapping region, allow overlap
                regions.append(self._generate_random_region(h, w))

        # Randomly choose which regions stay intact
        intact_indices = random.sample(range(len(regions)), self.num_regions_to_show)
        intact_regions = [regions[idx] for idx in intact_indices]

        # Create mask: start with all black
        masked_array = np.zeros_like(img_array)

        # Copy the intact regions from original image
        for intact_region in intact_regions:
            y1, y2, x1, x2 = intact_region
            masked_array[y1:y2, x1:x2] = img_array[y1:y2, x1:x2]

        # Save if output path provided
        if output_path:
            result_img = Image.fromarray(masked_array.astype(np.uint8))
            result_img.save(output_path)

        return masked_array, intact_regions

    def transform_with_visualization(self, image_path, output_path=None):
        """
        Transform image and return visualization showing the intact regions.

        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)

        Returns:
            dict: Contains 'transformed', 'intact_regions', and 'original' arrays
        """
        img = Image.open(image_path)
        original_array = np.array(img)

        transformed_array, intact_regions = self.transform(image_path, output_path)

        return {
            'original': original_array,
            'transformed': transformed_array,
            'intact_regions': intact_regions
        }


def main():
    """Test the ImageRegionMasker with a real image file."""
    import sys
    import os

    # Check if image path is provided as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Prompt user for image path
        print("Please provide the path to your image file:")
        print("Usage: python script.py <image_path>")
        print("\nOr enter the image path now:")
        image_path = input().strip()

    # Validate image path
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        print("Please provide a valid image file path.")
        return

    print(f"\nLoading image from: {image_path}")

    try:
        # Load and display image info
        img = Image.open(image_path)
        img_array = np.array(img)
        print(f"Image loaded successfully!")
        print(f"Image size: {img.size[0]}x{img.size[1]} pixels")
        print(f"Image shape: {img_array.shape}")
        print(f"Image mode: {img.mode}")

        # Test 1: Basic transformation
        print("\n" + "=" * 60)
        print("--- Test 1: Show 1 region (8 masked) ---")
        print("=" * 60)
        masker = ImageRegionMasker(num_regions_to_show=1, num_masked_regions=8)
        result, intact_regions = masker.transform(image_path, 'masked_output_1.png')

        print(f"Number of intact regions: {len(intact_regions)}")
        for i, (y1, y2, x1, x2) in enumerate(intact_regions, 1):
            print(f"  Region {i}: top={y1}, bottom={y2}, left={x1}, right={x2}")
            print(f"  Region {i} size: {y2 - y1}x{x2 - x1} pixels")
        print("✓ Output saved as 'masked_output_1.png'")

        # Test 2: Show multiple regions
        print("\n" + "=" * 60)
        print("--- Test 2: Show 3 regions (10 masked) ---")
        print("=" * 60)
        masker2 = ImageRegionMasker(num_regions_to_show=3, num_masked_regions=10, min_region_size=0.15,
                                    max_region_size=0.4)
        result2, intact_regions2 = masker2.transform(image_path, 'masked_output_2.png')
        print(f"Number of intact regions: {len(intact_regions2)}")
        for i, (y1, y2, x1, x2) in enumerate(intact_regions2, 1):
            print(f"  Region {i}: ({y1}, {y2}, {x1}, {x2}) - size: {y2 - y1}x{x2 - x1} pixels")
        print("✓ Output saved as 'masked_output_2.png'")

        # Test 3: With visualization
        print("\n" + "=" * 60)
        print("--- Test 3: Show 5 regions with analysis (12 masked) ---")
        print("=" * 60)
        masker3 = ImageRegionMasker(num_regions_to_show=5, num_masked_regions=12, min_region_size=0.15,
                                    max_region_size=0.35)
        results = masker3.transform_with_visualization(image_path, 'masked_output_3.png')

        y1, y2, x1, x2 = results['intact_regions'][0]
        print(f"Original shape: {results['original'].shape}")
        print(f"Transformed shape: {results['transformed'].shape}")
        print(f"Number of intact regions: {len(results['intact_regions'])}")
        for i, (y1, y2, x1, x2) in enumerate(results['intact_regions'], 1):
            print(f"  Region {i}: ({y1}, {y2}, {x1}, {x2}) - size: {y2 - y1}x{x2 - x1} pixels")
        print("✓ Output saved as 'masked_output_3.png'")

        # Verify masking
        print("\n" + "=" * 60)
        print("--- Verification Statistics ---")
        print("=" * 60)
        if len(results['transformed'].shape) == 3:
            black_pixels = np.sum(np.all(results['transformed'] == 0, axis=2))
        else:
            black_pixels = np.sum(results['transformed'] == 0)
        total_pixels = results['transformed'].shape[0] * results['transformed'].shape[1]
        intact_pixels = sum((y2 - y1) * (x2 - x1) for y1, y2, x1, x2 in results['intact_regions'])

        print(f"Total pixels: {total_pixels:,}")
        print(f"Black pixels: {black_pixels:,} ({100 * black_pixels / total_pixels:.1f}%)")
        print(f"Intact pixels: {intact_pixels:,} ({100 * intact_pixels / total_pixels:.1f}%)")

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - masked_output_1.png (1 visible region, 8 masked)")
        print("  - masked_output_2.png (3 visible regions, 10 masked)")
        print("  - masked_output_3.png (5 visible regions, 12 masked)")
        print("\nYou can customize by creating your own masker:")
        print("  masker = ImageRegionMasker(num_regions_to_show=2, num_masked_regions=6)")
        print("  masker.transform('your_image.jpg', 'output.png')")

    except Exception as e:
        print(f"\nError processing image: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
