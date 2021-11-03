#!/bin/bash

# Script to run experiments - loops through the text file of 
# random seeds training a model for each one. Saves the best
# model checkpoint (by validation loss) and the full confusion
# matrix, to be collated for analysis later in a jupyter 
# notebook.

for i in {1..25}
do
    python main.py \
    --experiment_name=EXP-R-${i}-LR001-WD00001 \
    --gpus=1 \
    --save_cm \
    --max_epochs=100 \
    --learning_rate=0.001 \
    --wd_param=0.00001 \
    --use_sds \
    --test \
    --optimizer=reduce
   
done