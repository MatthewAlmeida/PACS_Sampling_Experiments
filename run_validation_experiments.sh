#!/bin/bash

# This shell script loops through lines of the experiment manifest and 
# trains one model for each line of CLI arguments.

while read params; do
    python main.py $params
done < validation_experiment_manifest.txt