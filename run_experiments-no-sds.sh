#!/bin/bash

# Script to run experiments - loops through the text file of 
# random seeds training a model for each one. Saves the best
# model checkpoint (by validation loss) and the full confusion
# matrix, to be collated for analysis later in a jupyter 
# notebook.

while read seed; do
    python main.py \
    --experiment_name="Exp" \
    --random_seed=$seed \
    --gpus=1 \
    --no_logging \
    --save_cm \
    --max_epochs=75 
done < seeds_final.txt