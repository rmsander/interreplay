#!/bin/bash

# Change to user directory
cd ~

# Activate conda environment
conda init bash
source ~/.bashrc
conda activate interreplay

# Make new folder there
TMPFILE=`mktemp XXXXXXXXXX`
mkdir /state/partition1/user/$TMPFILE

# Copy mujoco-py folder to locked part of cluster
cp -r ~/mujoco-py /state/partition1/user/$TMPFILE/
cd /state/partition1/user/$TMPFILE/mujoco-py

# Now install it and import it to build
python3 setup.py install
python3 -c "import mujoco_py"

# Now move code to this folder and mujoco-py into code
cp -r ~/interreplay /state/partition1/user/$TMPFILE/
cp -r mujoco_py ../interreplay/

# Change direcrory to interreplay
cd ../interreplay

# Run code!  (With parameters
python3 cluster_run.py --input_path $1

# Remove temporary directory
rm -rf /state/partition1/user/$TMPFILE