#!/bin/bash

for dataset_config in digits pima swiss covertype breast eeg
do
    echo "Dataset ${dataset_config}"
    python experiments/bayesian_lr.py \
        configs/bayesian_log_reg.yaml \
        --dataset_config configs/datasets/${dataset_config}.yaml
done
