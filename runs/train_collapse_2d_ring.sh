#!/usr/bin/env bash

# python3 scripts/train_fc_synth.py   --data_type 2d_ring \
#                                     --disc_lr 1e-4 \
#                                     --k_d 2 \
#                                     --n_layers_d 1 \
#                                     --n_epochs 1000 \
#                                     --save_name n1_l1

# python3 scripts/train_fc_synth.py   --data_type 2d_ring \
#                                     --disc_lr 1e-3 \
#                                     --k_d 1 \
#                                     --n_layers_d 2 \
#                                     --n_epochs 1000 \
#                                     --save_name n5_l1 \
#                                     --batch_size 1024 \
#                                     --num_epoch_for_save 10

python3 scripts/train_fc_synth.py   --data_type 2d_ring \
                                    --disc_lr 1e-4 \
                                    --k_d 10 \
                                    --sigma 0.01 \
                                    --n_layers_d 2 \
                                    --n_epochs 1000 \
                                    --save_name n10_l2 \
                                    --batch_size 256 \
                                    --num_epoch_for_save 10

# python3 scripts/train_fc_synth.py   --data_type 2d_ring \
#                                     --disc_lr 1e-4 \
#                                     --k_d 2 \
#                                     --n_layers_d 2 \
#                                     --n_epochs 1000 \
#                                     --save_name n1_l2

# python3 scripts/train_fc_synth.py   --data_type 2d_ring \
#                                     --disc_lr 1e-4 \
#                                     --k_d 5 \
#                                     --n_layers_d 2 \
#                                     --n_epochs 1000 \
#                                     --save_name n5_l2