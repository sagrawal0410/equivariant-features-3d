#!/bin/bash

# Arguments are:
# $1: output directory name
# The rest of the arguments are the command to run the python script

# Throw an error if the output directory already exists
if [ -d outputs/$1 ]; then
    echo "Output directory already exists. Please delete it before running this script or choose another name."
    exit 1
fi

# Create the output directory
mkdir -p outputs/$1

# Copy all the workspace files into the output directory/workspace, excluding ./data and ./outputs and .pkl files lurking in the top level of the workspace
mkdir -p outputs/$1/workspace
rsync --quiet -av --exclude='outputs' --exclude='data' --exclude='*.pkl' --exclude='*.parquet' . outputs/$1/workspace --exclude='*.pth'

# Make a env.yaml file for the current conda environment
conda env export > outputs/$1/env.yaml

# Also make sure to include any conda rollback information in another txt file
conda list --revisions > outputs/$1/conda_rollback.txt

# Save the command to run the python script
echo $@ > outputs/$1/command.txt

# Run the python script form this new copied workspace directory
cd outputs/$1/workspace

# Run the python scripts and save the output and error logs, making sure to be aware of the fact that $1 might be a relative directory
shift
echo "Running the following command: $@"
eval "$@"
# > ../output.log 2> ../error.log"

