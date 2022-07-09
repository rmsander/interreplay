#!/bin/bash

# Read in all data from target directory
rsync -av -e ssh --exclude='*replay_buffer.pickle' --exclude='*checkpoint*' $2@txe1-login.mit.edu:/home/gridsan/$2/$1 $3/$1

