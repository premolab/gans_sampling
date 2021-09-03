Experiments based on papers https://arxiv.org/abs/1811.11357, https://arxiv.org/abs/2003.06060.


Implemented
[x] Version of i SIR and correlated i SIR in simple setting
[x] Toy experients on Gaussians
[x] correlated AI SIR
[x] Different ULA and MALA samplers
[x] Experiments on MCMC GAN stuff

To Do
[ ] Develop Ex^2 MCMC with part i-SIR and part MALA for refreshment kernel
[ ] Implement more enhanced version of correlation (Achille to write on overleaf)
[ ] Put Adaptive i-SIR in pipeline from flow and adaptive_mc files
[ ] Implement examples as in SA MCMC

To complete while we advance to NeurIPS !! :)

* to do fair ESS

# Installation

## With docker
Build docker image

```bash
docker build --tag ex2mcmc:v1 .
```

Run docker container

```bash
docker run -dit --name ex2mcmc ex2mcmc:v1
```

## Or without docker

```bash
conda create -y --name ex2mcmc python=3.8

conda activate ex2mcmc
```

```bash
pip install poetry
```

```bash
poetry install
```



# Usage
