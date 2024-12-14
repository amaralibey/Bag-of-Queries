#!/bin/bash


# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
# 
# This script is part of OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get the project root directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if Kaggle is installed
if ! command_exists kaggle; then
    echo "Kaggle is not installed. Please install Kaggle using: pip install kaggle"
    exit 1
fi

# Set the dataset name and download path
dataset_name="amaralibey/gsv-cities" # dataset name on Kaggle
download_path="$PROJECT_ROOT/data/train/gsv-cities" # path to download the dataset

# Create the download directory if it doesn't exist
mkdir -p "$download_path"

# make sur the folders Dataframes and Images do not exist, otherwise the dataset has already been downloaded
if [ -d "$download_path/Dataframes" ] || [ -d "$download_path/Images" ]; then
    echo "GSV-Cities dataset already downloaded to $download_path."
    exit 0
fi

# Download the dataset
echo "Downloading gsv-cities dataset from Kaggle..."
echo "To path: $download_path"
kaggle datasets download "amaralibey/gsv-cities" -p "$download_path" --unzip

if [ $? -eq 0 ]; then
    echo "GSV-Cities downloaded successfully to $download_path."
else
    echo "Error downloading dataset."
fi


echo "Download completed. Dataset available at $download_path."