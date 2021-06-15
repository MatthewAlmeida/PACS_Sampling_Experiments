#!/bin/bash

# SEED RANDOM with PID
RANDOM=$$

# Generate 100 random seeds from 
# 1 to 10,000

for i in `seq 100`
do
    R=$(($(($RANDOM%10000)) + 1))
    echo $R >> seeds.txt
done
