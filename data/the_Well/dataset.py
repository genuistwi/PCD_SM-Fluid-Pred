from fileinput import filename

import torch

from einops import rearrange
from the_well.data import WellDataset

from utils.files.utils import *

import lightning as lt


class theWellTorchDataset(lt.LightningDataModule):
    """
    Customized theWell dataset object.
    See https://huggingface.co/polymathic-ai for more details.
    """

    def __init__(self, work_dir, dataset_name, input_len, output_len, kind):

        super().__init__()
        # --- common attributes --- MUST BE SHARED AMONG DATASETS
        self.RAM_data = {}
        self.RAM_load_flag = False
        self.work_dir = work_dir
        self.cluster_path = os.path.join(self.work_dir, "data/_cluster_/")
        self.dataset_name = dataset_name


        assert dataset_name in ["turbulent_radiative_layer_2D",
                                "post_neutron_star_merger",
                                "viscoelastic_instability",
                                "MHD_64"]

        assert kind in ["train", "valid", "test", ]

        self.data_path = os.path.join(self.work_dir, "data/_pipeline_/datasets")
        self.kind = kind


        # --- Dataset ---
        self.data = None
        self.data_len = None
        self.data_keys = None

        self.fields_names = None
        self.fields_len = None  # Number of fields (= number of data input channels)

        self.input_len, self.output_len = input_len, output_len

        self.datasets_kwargs = dict(well_base_path=self.data_path, well_dataset_name=self.dataset_name,
                                    n_steps_input=self.input_len, n_steps_output=self.output_len, )

        # --- properties ---
        self.data_channels = None
        self.cond_channels = None


        # --- Normalization components (mean and std) ---
        if self.dataset_name == "turbulent_radiative_layer_2D":  # 2-dimensional, (T, x, y, F)
            """ 
            101 time steps so use 100 inputs and 1 output length to get a full time representation.
            Field names: ['density', 'pressure', 'velocity_x', 'velocity_y']. 
            Data keys: ["input_fields", "constant_scalars", "boundary_conditions", "space_grid", "input_time_grid"]. 
            """
            # -> True for train data only:
            self.mu = torch.tensor([3.4847e+01, 9.4475e-01, 6.1707e-03, -2.4651e-02])
            self.std = torch.tensor([4.4284e+01, 6.0970e-02, 4.1764e-02, 4.0095e-02])

        if self.dataset_name == "MHD_64":
            """
            100 time steps so use 99 inputs and 1 output length to get a full time representation.
            fields ['density', 'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z', 'velocity_x', 'velocity_y', 'velocity_z']
            ['input_fields', 'output_fields', 'constant_scalars', 'boundary_conditions', 'space_grid', 'input_time_grid', 'output_time_grid']
            """
            self.mu = torch.tensor([1.0015e+00, 5.3329e-01, 8.7613e-10, 1.9945e-08, 9.5950e-03, 6.7743e-02, 1.9342e-02])
            self.std = torch.tensor([0.8922, 0.5287, 0.3171, 0.3201, 0.4852, 0.4381, 0.4785])

        # if self.dataset_name == "post_neutron_star_merger":
        #     """ -> True for train data only.
        #     20 time steps so use 19 inputs and 1 output length to get a full time representation.
        #     Field names: ['density', 'pressure', 'velocity_x', 'velocity_y'].
        #     Data keys: ["input_fields", "constant_scalars", "boundary_conditions", "space_grid", "input_time_grid"].
        #     """
        #     # -> True for train data only:
        #     self.mu = torch.tensor([3.4847e+01, 9.4475e-01, 6.1707e-03, -2.4651e-02])
        #     self.std = torch.tensor([4.4284e+01, 6.0970e-02, 4.1764e-02, 4.0095e-02])
        #
        # if self.dataset_name == "viscoelastic_instability":  # 3-dimensional, (T, x, y, z, F)
        #     """
        #     20 time steps so use 19 inputs and 1 output length to get a full time representation.
        #     Field names:
        #     ['density', 'internal_energy', 'pressure', 'temperature', 'electron_fraction', 'entropy',
        #     'magnetic_field_log_r', 'magnetic_field_theta', 'magnetic_field_phi',
        #     'velocity_log_r', 'velocity_theta', 'velocity_phi'].
        #     Data keys: ["input_fields", "constant_scalars", "boundary_conditions", "space_grid", "input_time_grid"].
        #     """
        #     # -> True for train data only:
        #     self.mu = torch.tensor([3.4847e+01, 9.4475e-01, 6.1707e-03, -2.4651e-02])
        #     self.std = torch.tensor([4.4284e+01, 6.0970e-02, 4.1764e-02, 4.0095e-02])

        # --- Intern components ---
        self.load_flag = False
        self.reshape_flag = False
        self.normalize_flag = False

        # --- Intern operations ---
        self.custom_operations = None

    def load(self):

        self.data = WellDataset(well_split_name=self.kind, use_normalization=False, **self.datasets_kwargs, )
        self.data_len = len(self.data)

        self.data_keys = list(self.data[0].keys())

        self.fields_names = [name for group in self.data.metadata.field_names.values() for name in group]
        self.fields_len = len(self.fields_names)

        self.load_flag = True


    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        if not self.RAM_load_flag:
            input_data = self.custom_operations(self.data[idx]["input_fields"])
            output_data = self.custom_operations(self.data[idx]["output_fields"])
        else:
            input_data, output_data = self.RAM_data[idx]

        return input_data, output_data

    def set_custom_flags(self, reshape=True, normalize=True, subsampling_factor=1):
        self.reshape_flag = True if reshape else False
        self.normalize_flag = True if normalize else False
        sf = subsampling_factor

        resh_only = self.reshape_flag and not self.normalize_flag
        norm_only = self.normalize_flag and not self.reshape_flag
        resh_norm = self.reshape_flag and self.normalize_flag

        # 2-dimensional kind
        if self.dataset_name in ["turbulent_radiative_layer_2D"]:
            shape_str = "T Lx Ly F -> T F Lx Ly"  # Note: When reshaped T F Lx Ly, T is increasing forward.
            normalize_fn = lambda X: (X - self.mu[None, None, None, :]) / self.std[None, None, None, :]

            if resh_only:
                self.custom_operations = lambda sample: rearrange(sample[:,::sf,::sf,:], shape_str)
            elif norm_only:
                self.custom_operations = lambda sample: normalize_fn(sample[:,::sf,::sf,:])
            elif resh_norm:
                self.custom_operations = lambda sample: rearrange(normalize_fn(sample[:,::sf,::sf,:]), shape_str)
            else:
                self.custom_operations = lambda sample: sample[:,::sf,::sf,:]

        if self.dataset_name in ["MHD_64"]:
            shape_str = "T Lx Ly Lz F -> T F Lx Ly Lz"  # Note: When reshaped T F Lx Ly, T is increasing forward.
            normalize_fn = lambda X: (X - self.mu[None, None, None, None, :]) / self.std[None, None, None, None, :]

            if resh_only:
                self.custom_operations = lambda sample: rearrange(sample[:,::sf,::sf,::sf,:], shape_str)
            elif norm_only:
                self.custom_operations = lambda sample: normalize_fn(sample[:,::sf,::sf,::sf,:])
            elif resh_norm:
                self.custom_operations = lambda sample: rearrange(normalize_fn(sample[:,::sf,::sf,::sf,:]), shape_str)
            else:
                self.custom_operations = lambda sample: sample[:,::sf,::sf,::sf,:]


    def init(self, kwargs):
        self.load()
        self.set_custom_flags(**kwargs)

        self.data_channels = self.output_len
        self.cond_channels = self.data_channels



