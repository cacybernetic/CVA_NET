import os
import shutil
import argparse

def merge_datasets(dataset1, dataset2, merged_dataset):
    """
    Merges two image classification datasets into a new directory.
    Handles duplicate filenames by appending suffixes.
    """
    # Create the merged dataset directory if it doesn't exist
    os.makedirs(merged_dataset, exist_ok=True)

    # Collect all labels from both datasets
    labels = set()
    for dataset in [dataset1, dataset2]:
        if os.path.exists(dataset):
            labels.update(d for d in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, d)))

    for label in labels:
        label_dir = os.path.join(merged_dataset, label)
        os.makedirs(label_dir, exist_ok=True)
        existing_files = set(os.listdir(label_dir))

        # Process each dataset for the current label
        for dataset in [dataset1, dataset2]:
            src_label_dir = os.path.join(dataset, label)
            if not os.path.isdir(src_label_dir):
                continue

            for filename in os.listdir(src_label_dir):
                src_path = os.path.join(src_label_dir, filename)
                if not os.path.isfile(src_path):
                    continue

                # Handle duplicate filenames
                base_name, ext = os.path.splitext(filename)
                dst_filename = filename
                counter = 1
                while dst_filename in existing_files:
                    dst_filename = f"{base_name}_{counter}{ext}"
                    counter += 1

                # Copy the file to the merged directory
                dst_path = os.path.join(label_dir, dst_filename)
                shutil.copy2(src_path, dst_path)
                existing_files.add(dst_filename)
                print(f"Copied: {src_path} -> {dst_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge two image classification datasets.")
    parser.add_argument('-d1', "--dataset1", required=True, help="Path to the first dataset directory")
    parser.add_argument('-d2', "--dataset2", required=True, help="Path to the second dataset directory")
    parser.add_argument('-md', "--merged_dataset", required=True, help="Path to the merged output directory")
    args = parser.parse_args()

    # Validate input directories
    if not os.path.isdir(args.dataset1):
        raise ValueError(f"Dataset1 directory '{args.dataset1}' does not exist.")
    if not os.path.isdir(args.dataset2):
        raise ValueError(f"Dataset2 directory '{args.dataset2}' does not exist.")

    merge_datasets(args.dataset1, args.dataset2, args.merged_dataset)
    print("Datasets merged successfully!")

if __name__ == "__main__":
    main()
