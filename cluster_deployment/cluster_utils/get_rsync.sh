#!/bin/bash

# Read in all data from target directory
rsync -av -e ssh --exclude='*replay_buffer.pickle' --exclude='*checkpoint*' $1 $2

