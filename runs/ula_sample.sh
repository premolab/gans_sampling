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

python3 scripts/gaussians_mcmc.py \
            --gan_data_dir $1 \
            --model_idx 35 40 45 50 55 60 70 80 90 \
            --sampler citerais_ula \
            --calibrate \
            --save_prefix calib_mala \
            --grad_step 1e-4 \
            --eps_scale 1e-4 \
            --rho 0.95 \
            --T 150 \
            --n_steps 200 \
            --batch_size 10 \
            --mala_afterall \
            --beta_deg 0.1

python3 scripts/gaussians_mcmc.py \
            --gan_data_dir $1 \
            --model_idx 35 40 45 50 55 60 70 80 90 \
            --sampler citerais_ula ula \
            --calibrate \
            --save_prefix calib \
            --grad_step 1e-4 \
            --eps_scale 1e-4 \
            --rho 0.95 \
            --T 150 \
            --n_steps 200 \
            --batch_size 10 \
            --beta_deg 0.1

python3 scripts/gaussians_mcmc.py \
            --gan_data_dir $1 \
            --model_idx 35 40 45 50 55 60 70 80 90 \
            --sampler citerais_ula \
            --save_prefix mala \
            --grad_step 1e-4 \
            --eps_scale 1e-4 \
            --rho 0.95 \
            --T 150 \
            --n_steps 200 \
            --batch_size 10 \
            --mala_afterall \
            --beta_deg 0.1

python3 scripts/gaussians_mcmc.py \
            --gan_data_dir $1 \
            --sampler citerais_ula ula \
            --save_prefix new \
            --grad_step 1e-4 \
            --eps_scale 1e-4 \
            --rho 0.95 \
            --T 150 \
            --n_steps 200 \
            --batch_size 10 \
            --calibrate \
            --model_idx 35 40 45 50 55 60 70 80 90 \
            --beta_deg 0.1