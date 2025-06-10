import os
import sys
import types

from datetime import datetime
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_r0(message): print(message)


def get_current_datetime(): return datetime.now().strftime("y%Y_m%m_d%d_%Hh_%Mm_%Ss")


class ConfigParser:
    """
    A class to dynamically load and parse Python `.py` config files.
    This class allows loading Python config files as modules from a given path and
    accessing their objects using a specified name. Access these objects using dot notation.
    Also supports complex object configurations.

    Also acts as a global object for code configuration/interpretation and self saving/exporting.
    """

    def __init__(self, work_dir: str):

        self.work_dir = os.path.join(os.getcwd(), work_dir)

        self.date_time = None
        self.ID = None
        self.ckpt_dir = None

        self.load_flag = False
        self.cluster = False

        # --- custom attributes for code ---
        self.optimizer = None
        self.EMA = None
        self.model = None
        self.sde = None
        self.input_format = None
        self.cond_format = None

        # --- custom objects for training ---
        self.losses = []
        self.step = 0
        self.batch_loss = None
        self.batch_idx = 0
        self.epoch_idx = 0
        self.MSE_ref = None
        self.MSE_est = None


        # --- sub configs ---
        self.project_name = None  # WandB only
        self.globalCfg = None
        self.modelCfg = None
        self.sdeCfg = None
        self.trainingCfg = None
        self.samplingCfg = None


    def start(self):
        self.date_time = get_current_datetime()
        self.ID = self.date_time
        self.ckpt_dir = os.path.join(self.work_dir, "storage/models/" + self.ID + "/")


    def load_config(self, config_path: str, config_name: str) -> None:
        config_path = os.path.join(self.work_dir, config_path)
        with open(config_path, "r") as file: code = file.read()
        local_scope = {}
        exec(code, {}, local_scope)
        config_obj = types.SimpleNamespace(**local_scope)
        setattr(self, config_name, config_obj)

    def load(self, config_file: str) -> None:
        """
        Loads multiple config files specified in a text file.
        Starts multiprocess if possible.
        """

        if not os.path.exists(config_file): raise ValueError(f"Config list file not found: {config_file}")

        with open(config_file, 'r') as f:

            for file_counter, line in enumerate(filter(lambda l: l.strip() and not l.startswith("#"), f)):
                parts = line.strip().split()
                if len(parts) != 2: raise ValueError(f"Invalid line format: '{line.strip()}'")

                path, name = parts
                file_name = path.rsplit("/", 1)[-1]

                if file_counter == 0 and file_name != "global.py":
                    print(f"Parser must start reading at globalCfg.py, adjust {config_file}.")
                    sys.exit(1)
                try:
                    self.load_config(path, name)
                    if file_counter == 0: self.start()  # Start if globalCfg has been read
                    print_r0(f"Loaded config '{name}' from '{path}'")
                except Exception as e:
                    print(f"Error loading config '{name}' from '{path}': {e}")

        self.load_flag = True
        self.project_name = self.modelCfg.model_name + "_" + self.globalCfg.dataset_name
