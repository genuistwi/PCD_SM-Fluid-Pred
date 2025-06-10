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

Run **`data/JHTDB/clustering.ipynb`** and **`data/the_Well/clustering.ipynb`** for data processing.

### _Training and sampling_

Train a model by running **`training.py`**.
Dataset choice and hyperparameter values can be changed in the config directory.
Models are saved in **`storage/models`** with a unique ID (data+time).

Sampling can be launched by running **`sampling.py`** and specifying a list of IDs (trained models).

Pre-trained models used for the paper can be downloaded at https://nextcloud.centralesupelec.fr/f/10550179, 
https://nextcloud.centralesupelec.fr/f/10550173 and https://nextcloud.centralesupelec.fr/f/10550188.
(3`.zip` files, paste model/ID directories in **`storage/models.py`**).
A list of associated IDs can be found in **`results/<dataset_name>/Notes.md`**.

Graphs and tables can be obtained in **`results/<dataset_name>/graphs.ipynb`** (once sampling is done).

