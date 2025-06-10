# Autoregressive regularized score-based diffusion models for multi-scenarios fluid flow prediction

**Credits**: Wilfried GENUIST | **Contact**: wilfried.genuist@centralesupelec.fr | **Preprint**: https://arxiv.org/abs/2505.24145.


## <u>Installation details</u> (tested on Linux-Ubuntu 22.+, python 3.10)

- Install libraries via conda:
```
conda env create -f environment.yml
```
- Activate environment:
```
source activate SM-Fluid-Pred
```

## <u>Testing and running</u>

### _Downloading data and formatting_

Dataset can be automatically downloaded by running **`data/JHTDB/download.ipynb`**, **`data/the_Well/download.ipynb`**.

Run **`data/JHTDB/clustering.ipynb`** and **`data/the_Well/clustering.ipynb`** for data process.
