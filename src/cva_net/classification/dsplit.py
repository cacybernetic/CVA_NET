#!/usr/bin/env python3
# -*- encoding: utf8 -*-

"""
+==============================================================================+
|                     CLASSIFICATION DATASET SPLITTER                          |
+==============================================================================+

Here is input dataset condition:
dataset_dir
  label1
    sample_file1
    sample_file2
    sample_file3
    sample_file4
    ...
  label2
    sample_file1
    sample_file2
    sample_file3
    ...
  ...
"""

__version__ = '0.1.0'
__author__ = 'DOCTOR MOKIRA'

import os
import random
import time as tm
import typing as t
from shutil import copy
from argparse import ArgumentParser


def print_progress(message: str, progress: int, total: int):
    print("\033[2K", end='\r')
    print(message, f"{progress}/{total}", end=' ', flush=True)


def find_min_sample_count(folder: str, classes: t.List[str]) -> int:
    """
    Function to find the mininal number of samples can be found.

    :param folder: The path to dataset directory.
    :param classes: The list of the class names found.
    :returns: The minimal number of samples.
    """
    min_samples_count = float('+inf')
    num_classes = len(classes)
    for index, class_name in enumerate(classes):
        cn_fp = os.path.join(folder, class_name)
        sample_files = os.listdir(cn_fp)
        if min_samples_count <= len(sample_files):
            continue
        min_samples_count = len(sample_files)
        print_progress("Min sample count finding ... ", index + 1, num_classes)
    return min_samples_count


def main() -> int:
    """
    Main function to run dataset splitting.
    """
    parser = ArgumentParser(prog="Dataset Splitter")
    parser.add_argument("input", type=str, help="Dataset directory to split.")
    parser.add_argument("output", type=str, help="Output directory of ds.")
    parser.add_argument("--test", type=float, default=0.9)
    args = parser.parse_args()

    input_dir = args.input
    train_dir = os.path.join(args.output, "train")
    test_dir = os.path.join(args.output, "test")
    test_proportion = args.test

    if not os.path.isdir(input_dir):
        print(f"No such dataset directory at \"{input_dir}\".")
        return 2

    # Create dataset output directory
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get available classes.
    # If no classes found, exit program.
    class_names = os.listdir(input_dir)
    if not class_names:
        print(f"Empty dataset on {input_dir}.")
        return 1

    # Find minimal samples count can be found:
    min_samples_count = find_min_sample_count(input_dir, class_names)
    num_classes = len(class_names)
    num_samples = int(test_proportion * min_samples_count)

    for idx, class_name in enumerate(class_names):
        # Load all samples contained in this class:
        class_name_fp = os.path.join(input_dir, class_name)
        samples = os.listdir(class_name_fp)

        # Select num_samples samples randomly:
        random.shuffle(samples)
        test_samples = samples[:num_samples]
        train_samples = samples[num_samples:]

        # Create current class dir in output directory:
        train_cl_fp = os.path.join(train_dir, class_name)
        test_cl_fp = os.path.join(test_dir, class_name)
        os.makedirs(train_cl_fp, exist_ok=True)
        tm.sleep(0.01)
        os.makedirs(test_cl_fp, exist_ok=True)
        tm.sleep(0.01)

        # Copy all samples selected into corresponding class directory
        # previously created in output directory.
        for selected_sample in train_samples:
            inp_sample_fp = os.path.join(class_name_fp, selected_sample)
            out_sample_fp = os.path.join(train_cl_fp, selected_sample)
            copy(inp_sample_fp, out_sample_fp)
            tm.sleep(0.01)

        for selected_sample in test_samples:
            inp_sample_fp = os.path.join(class_name_fp, selected_sample)
            out_sample_fp = os.path.join(test_cl_fp, selected_sample)
            copy(inp_sample_fp, out_sample_fp)
            tm.sleep(0.01)

        print_progress(
            message=f"Process class name: '{class_name}' ...",
            progress=(idx + 1), total=num_classes
        )


if __name__ == '__main__':
    code = main()
    exit(code)
