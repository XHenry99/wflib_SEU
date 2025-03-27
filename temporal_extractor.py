import numpy as np
import os
import argparse
from tqdm import tqdm
from WFlib.tools import data_processor

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Temporal feature extraction of Holmes')

# Define command-line arguments
parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the npz files")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
parser.add_argument("--in_file", type=str, default="train", help="Input file name. If 'all', process all npz files in the directory")

# Parse command-line arguments
args = parser.parse_args()

def extract_temporal_feature(X, feat_length=1000):
    # 确保 X 是 numpy 数组
    X = np.array(X)
    abs_X = np.absolute(X)
    new_X = []

    for idx in tqdm(range(X.shape[0])):
        temporal_array = np.zeros((2, feat_length))
        loading_time = abs_X[idx].max()
        interval = 1.0 * loading_time / feat_length

        for packet in X[idx]:
            if packet == 0:
                break
            elif packet > 0:
                order = int(packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[0][order] += 1
            else:
                order = int(-packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[1][order] += 1
        new_X.append(temporal_array)
    new_X = np.array(new_X)
    return new_X

def process_npz_file(npz_file):
    # Construct the output file path
    file_name = os.path.basename(npz_file)
    out_file = os.path.join(os.path.dirname(npz_file), f"temporal_{os.path.splitext(file_name)[0]}.npz")

    # Check if the output file already exists
    if not os.path.exists(out_file):
        # Load data from the specified input file
        X, y = data_processor.load_data(npz_file, "Origin", args.seq_len)

        # Extract temporal features from the input data
        temporal_X = extract_temporal_feature(X)

        # Print the shape of the extracted temporal features
        print("Shape of temporal_X:", temporal_X.shape)

        # Save the extracted features and labels to the output file
        np.savez(out_file, X=temporal_X, y=y)
    else:
        # If the output file already exists, print a message indicating it has been generated
        print(f"{out_file} has been generated.")

if args.in_file == "all":
    # Process all npz files in the directory
    for root, dirs, files in os.walk(args.dataset_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_file = os.path.join(root, file)
                process_npz_file(npz_file)
else:
    # Process a specific npz file
    npz_file = os.path.join(args.dataset_dir, f"{args.in_file}.npz")
    if os.path.exists(npz_file):
        process_npz_file(npz_file)
    else:
        print(f"The specified npz file {npz_file} does not exist.")