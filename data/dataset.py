import pickle
from tqdm import tqdm


def RAM_load(dataset):
    for sample_idx in tqdm(range(len(dataset))):
        dataset.RAM_data[sample_idx] = dataset[sample_idx]
    dataset.RAM_load_flag = True

def RAM_export(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def pkl_import(path):
    with open(path, "rb") as f:
        return pickle.load(f)

