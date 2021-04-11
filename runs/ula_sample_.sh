#~/usr/bin/env bash

# python3 scripts/gaussians_mcmc.py \
#             --gan_data_dir $1 \
#             --model_idx 5 10 20 25 35 40 45 50 55 60 70 80 90 \
#             --sampler citerais_ula ula

# python3 scripts/gaussians_mcmc.py \
#             --gan_data_dir $1 \
#             --model_idx 35 40 45 50 55 60 70 80 90 \
#             --sampler citerais_ula ula

# python3 scripts/gaussians_mcmc.py \
#             --gan_data_dir $1 \
#             --model_idx 5 10 20 25 35 40 45 50 55 60 70 80 90 \
#             --sampler citerais_ula \
#             --ula_afterall \
#             --save_prefix ula_after

# python3 scripts/gaussians_mcmc.py \
#             --gan_data_dir $1 \
#             --model_idx 5 10 20 25 35 40 45 50 55 60 70 80 90 \
#             --sampler citerais_ula \
#             --rho 0 \
#             --save_prefix rho0


python3 scripts/stacked_mnist_mcmc.py \
            --gan_data_dir $1 \
            --data_dir ../data \
            --model_idx 49 \
            --sampler ula \
            --calibrate \
            --save_prefix calib \
            --grad_step 1e-3 \
            --eps_scale 1e-3 \
            --rho 0.95 \
            --T 1 \
            --n_steps 250 \
            --batch_size 100 \
            --beta_deg 0.125 \
            --clf_path dump/StackedMNIST/mnist_clf.pth
