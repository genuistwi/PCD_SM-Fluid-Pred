import os
import pickle
import h5py


# Flag multiprocess

def path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
        return 0
    else:
        return 1


# Pickle save and load
def pkl_dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def pkl_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# h5py save and load
def h5py_dump(obj, path, name):
    with h5py.File(path, "w") as f:
        f.create_dataset(name, data=obj)

def h5py_load(path, name):
    with h5py.File(path, "r") as f:
        return f[name][:]

