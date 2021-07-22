#!/bin/bash

# Script to run experiments - loops through the text file of 
# random seeds training a model for each one. Saves the best
# model checkpoint (by validation loss) and the full confusion
# matrix, to be collated for analysis later in a jupyter 
# notebook.

for i in {1..100}
do
    python main.py \
    --experiment_name="AP-LR1e04-WP1e-3" \
    --gpus=1 \
    --save_cm \
    --use_sds \
    --max_epochs=100 \
    --learning_rate=0.0001 \
     --wd_param=0.001 \
    --test
done