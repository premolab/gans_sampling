## Usage

```bash
python experiments/complex_geometry.py configs/complex_geom.yaml --dist_config configs/dists/{funnel,banana}.yaml
```

function to compute metrics: ```complex_geometry.compute_metrics```


Config: 

```yaml
batch_size : 1 # number of independent chains
device: cpu
figpath: figs
data_root: data

trunc_chain_len: 500 # number of points to compute metrics

methods:
  # MALA:
  #   mcmc_class: MALA
  #   n_steps: 20000 #50000
  #   ess_rar: 20 #50
  #   params:
  #     grad_step: 0.25
  #     adapt_stepsize: True
  #   color : "tab:green"

  "Ex$^2$MCMC":
    mcmc_class: Ex2MCMC
    n_steps: 1000
    ess_rar: 1 # take every k'th point to compute ess fairly if number of steps differs for defferent methods
    params:
      N: 100 # number of particles
      grad_step: .5 # step size for rejuvenation kernel
      adapt_stepsize: False # adapt step size to balance acceptance rate
      corr_coef: 0.975 # correlation coef for particles 
      bernoulli_prob_corr: 0.5 # prob of correlation
      mala_steps: 3 # number of rejuvenation steps
    color : "tab:red"

  "FlEx$^2$MCMC":
    mcmc_class: Ex2MCMC
    n_steps: 1000
    ess_rar: 1
    params:
      N: 100
      grad_step: .5
      adapt_stepsize: True
      corr_coef: 0.975
      bernoulli_prob_corr: 0.5
      mala_steps: 3
    flow:
      num_flows: 6 # number of normalizing layers 
      lr: 0.005 # learning rate 
      batch_size: 200
      n_steps: 200
    color: 'tab:pink'

```


Distribution config:

```yaml
dist: funnel
dist_class: Funnel
scale_proposal: 1. # scale for normal proposal
params: {}
n_steps: 1 # meaningless
dims: [15] #[5, 15] #[5, 10, 15, 20]
```

