import os
import pickle
import dill
import pickle
import torch
import io

def save_objects(storage_path, sampleCfg, data_dict, test_case_number, additional_string=None):
    """
    Saves data_dict and sampleCfg in a structured directory:

    storage_path/
      └── results/
           └── sampleCfg.ID/
                └── <predictor>_ODE_<probability_flow>/
                     └── <version>/   # version: "1", "2", etc.
                           ├── dict.pkl
                           └── sampleCfg.pkl
    """
    # Build the main directory: storage_path/results/sampleCfg.ID
    main_dir = os.path.join(storage_path, "results/" + sampleCfg.globalCfg.dataset_name, )
    os.makedirs(main_dir, exist_ok=True)

    method = f"__ODE_{sampleCfg.samplingCfg.probability_flow}" + ("" if additional_string is None else additional_string)

    # Build the subdirectory name using predictor and probability_flow
    sub_dir_name = (f"E_loss_{str(sampleCfg.trainingCfg.energy_loss)}" +
                    method + "/" +
                    f"{sampleCfg.ID}" + "/" + "test_case_" + str(test_case_number))


    sub_dir = os.path.join(main_dir, sub_dir_name)
    os.makedirs(sub_dir, exist_ok=True)

    # Find a version number for the subsubdirectory ("1", "2", ...)
    version = 1
    while True:
        final_dir = os.path.join(sub_dir, str(version))
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            break
        version += 1

    # Save the dictionary as pickle
    dict_filename = os.path.join(final_dir, "dict.pkl")
    with open(dict_filename, "wb") as f:
        dill.dump(data_dict, f)

    # Save the sampleCfg object as pickle
    sampleCfg_filename = os.path.join(final_dir, "sampleCfg.pkl")
    with open(sampleCfg_filename, "wb") as f:
        dill.dump(sampleCfg, f)

    print(f"Data saved in directory: {final_dir}")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def safe_pickle_load_cpu(file_path):
    with open(file_path, 'rb') as f:
        return CPU_Unpickler(f).load()


def load_objects(storage_path, dataset_name, sub_dir_name, sampleCfg_id, test_case_number, version):
    """
    Loads and returns the objects (data_dict and sampleCfg) saved in the structured directory:

    storage_path/
      └── results/
           └── sampleCfg_id/
                └── sub_dir_name/       # format: "<predictor>_ODE_<probability_flow>"
                     └── version/      # version as string or integer, e.g., "1", "2", etc.
                           ├── dict.pkl
                           └── sampleCfg.pkl

    Parameters:
        storage_path (str): Base storage directory.
        sampleCfg_id (str): The ID associated with sampleCfg.
        sub_dir_name (str): The subdirectory name, e.g., "<predictor>_ODE_<probability_flow>".
        version (int or str): The version number subdirectory (e.g., 1, 2, etc.).

    Returns:
        tuple: (data_dict, sampleCfg) loaded from pickle files.

    Raises:
        FileNotFoundError: If the expected file structure does not exist.
    """
    # Construct the directory path
    base_dir = os.path.join(storage_path, "results/",
                            dataset_name, sub_dir_name, sampleCfg_id,
                            "test_case_" + str(test_case_number), str(version))

    # Define file paths
    dict_file = os.path.join(base_dir, "dict.pkl")
    sampleCfg_file = os.path.join(base_dir, "sampleCfg.pkl")

    if not os.path.exists(dict_file):
        raise FileNotFoundError(f"Dictionary file not found: {dict_file}")
    if not os.path.exists(sampleCfg_file):
        raise FileNotFoundError(f"sampleCfg file not found: {sampleCfg_file}")

    # Load the dictionary
    with open(dict_file, 'rb') as f:
        if torch.cuda.is_available():
            data_dict = pickle.load(f)  # First try the regular way
        else:
            print('CUDA err')
            data_dict = safe_pickle_load_cpu(dict_file)


    # Load the sampleCfg object
    with open(sampleCfg_file, "rb") as f:
        sampleCfg = pickle.load(f)

    return data_dict, sampleCfg
