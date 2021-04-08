#!/usr/bin/env bash

python3 scripts/train_fc_synth.py   --data_type 2d_grid \
                                    --disc_lr 1e-4 \
                                    --k_d 10 \
                                    --n_layers_d 2 \
                                    --batch_size 512 \
                                    --n_epochs 3000 \
                                    --sigma 0.02 \
                                    --num_epoch_for_save 10