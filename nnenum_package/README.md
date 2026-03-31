# nnenum stars quantized in latent dimension

## Setup

### conda

Set up the conda development environment called `latent-quantization`:

```bash
conda env create --file=environment.yml
```

Enable the environment:

```bash
conda activate latent-quantization
```

### nnenum

Install `nnenum` inside the conda environment:

```bash
cd nnenum
pip install .
cd ../
```

Set the following environment variables, required by `nnenum`:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Ensure `nnenum` works by testing some basic imports:

```bash
python -c "import nnenum.lp_star ; help(nnenum.lp_star.LpStar)"
```

### iq_verify

Install `iq_verify` inside the conda environment:

```bash
cd iq_verify
pip install .
cd ../
```

Ensure `iq_verify` works by testing some basic imports:

```bash
python -c "from iq_verify.set_repr import star_set ; help(star_set.StarSet)"
```

## Run the script

Plot the star sets and their quantized points, as seen in our Slack chat:

```bash
python main.py
```