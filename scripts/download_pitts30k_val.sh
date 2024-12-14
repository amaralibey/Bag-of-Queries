#!/bin/bash

# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
# 
# This script is part of OpenVPRLab: https://github.com/amaralibey/OpenVPRLab
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

# Get the project root directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set the dataset name and download path
DATASET_NAME="pitts30k-val"
DOWNLOAD_URL="https://github.com/amaralibey/OpenVPRLab/releases/download/v0.1/pitts30k-val.zip"
DESTINATION_PATH="$PROJECT_ROOT/data/val"

# Make sure the DESTINATION_PATH already exists
if ! [ -d "$DESTINATION_PATH" ]; then
    echo "The folder $DESTINATION_PATH does not exist. Please check!!"
    exit 1
fi

# make sure the dataset hasn't already been downloaded 
if [ -d "$DESTINATION_PATH/$DATASET_NAME" ]; then
    echo "The folder $DESTINATION_PATH/$DATASET_NAME exists, have you already downloaded the dataset? Please check $DESTINATION_PATH"
    exit 0
fi

# Download the dataset
echo "Downloading $DATASET_NAME dataset for OpenVPRLab..."
echo "Saving to path: $DESTINATION_PATH/$DATASET_NAME.zip"

if ! curl -L -o "$DESTINATION_PATH/$DATASET_NAME.zip" "$DOWNLOAD_URL"; then
    echo "Failed to download the dataset."
    exit 1
fi

# Unzip the downloaded zip file
echo "Download successful. Now unzipping $DATASET_NAME.zip..."
if ! unzip -q "$DESTINATION_PATH/$DATASET_NAME.zip" -d "$DESTINATION_PATH"; then
    echo "Failed to unzip the file."
    exit 1
fi

# Remove the zip file after successful extraction
if ! rm "$DESTINATION_PATH/$DATASET_NAME.zip"; then
    echo "Warning: Failed to remove zip file: $DESTINATION_PATH/$DATASET_NAME.zip"
fi

echo "Dataset successfully downloaded and extracted to $DESTINATION_PATH/$DATASET_NAME"
