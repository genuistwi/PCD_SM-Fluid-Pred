import sys

import torch
import pytorch_lightning as pl
from torch.utils.data import RandomSampler, random_split, DataLoader
# from data.dataset import Dataset
import os
from torch.utils.data import Dataset, Subset


# --- The Well data utils ---
from data.the_Well.dataset import theWellTorchDataset

# --- JHTDB data utils ---
from data.JHTDB.dataset import TurbulenceDataset
from data.JHTDB.params import DataParams
from data.JHTDB.transforms import Transforms

from data.dataset import RAM_export, RAM_load, pkl_import


theWell_subFact = 2


def TheWell_datasetInit(dataModule, export, cluster_import):
    assert dataModule.dataset_name in ["turbulent_radiative_layer_2D", "MHD_64"]

    formatKwargs = dict(input_len=1, output_len=1)
    theWellKwargs = dict(work_dir=dataModule.Cfg.work_dir, dataset_name=dataModule.dataset_name, **formatKwargs)
    dataModule.trainSet = theWellTorchDataset(**theWellKwargs, kind="train")
    dataModule.validSet = theWellTorchDataset(**theWellKwargs, kind="valid")

    # --- Needed for model init ---
    dataModule.trainSet.init(dict(subsampling_factor=theWell_subFact))
    dataModule.validSet.init(dict(subsampling_factor=theWell_subFact))

    # ---------- ---------- ---------- ---------- ----------
    cluster_path = dataModule.trainSet.cluster_path
    dataset_name = dataModule.dataset_name
    train_save_path = os.path.join(cluster_path, dataset_name + "_train.pkl")
    valid_save_path = os.path.join(cluster_path, dataset_name + "_valid.pkl")
    # --- RAM load ---
    if export:
        print("Manual export is preferable.")
        # RAM_load(dataModule.trainSet);  RAM_export(train_save_path, dataModule.trainSet.RAM_data)
        # del dataModule.trainSet
        # RAM_load(dataModule.validSet); RAM_export(valid_save_path, dataModule.validSet.RAM_data)
        # del dataModule.validSet
    # --- load exported data method ---
    if cluster_import:
        dataModule.trainSet.RAM_data = pkl_import(train_save_path)
        dataModule.validSet.RAM_data = pkl_import(valid_save_path)
        dataModule.trainSet.RAM_load_flag = True
        dataModule.validSet.RAM_load_flag = True
        print("Loaded datasets from ./data/_cluster_.")
    # ---------- ---------- ---------- ---------- ----------

def TheWell_test(dataModule):
    assert dataModule.dataset_name in ["turbulent_radiative_layer_2D", "MHD_64"]

    if dataModule.dataset_name == "turbulent_radiative_layer_2D":
        formatKwargs = dict(input_len=100, output_len=1)
    if dataModule.dataset_name == "MHD_64":
        formatKwargs = dict(input_len=99, output_len=1)

    theWellKwargs = dict(work_dir=dataModule.Cfg.work_dir, dataset_name=dataModule.dataset_name, **formatKwargs)
    dataModule.testSet = theWellTorchDataset(**theWellKwargs, kind="test")
    dataModule.testSet.init(dict(subsampling_factor=theWell_subFact))


def JHTDB_datasetInit(dataModule, export, cluster_import):
    nameKwargs = dict(name="Training", dataDirs=["data/_pipeline_/JHTDB"])
    filterKwargs = dict(filterTop=["128_tra"], excludefilterSim=True, filterFrame=[(0, 1000)])
    configKwargs = dict(randSeqOffset=True, printLevel="sim")
    fieldKwargs = dict(simFields=["dens", "pres"], simParams=["mach"])

    dataModule.trainSet = TurbulenceDataset(**nameKwargs, **filterKwargs, **fieldKwargs, **configKwargs,
                                            filterSim=[[0, 1, 2, 14, 15, 16, 17, 18]], sequenceLength=[[2, 2]],
                                            work_dir=dataModule.Cfg.work_dir)

    dataParamsKwargs = dict(augmentations=["normalize"], randSeqOffset=True, dimension=2)
    normaKwargs = dict(simFields=["dens", "pres"], simParams=["mach"], normalizeMode="machMixed")

    p_d = DataParams(**dataParamsKwargs, **normaKwargs, dataSize=[128, 64])

    transTrain = Transforms(p_d)
    dataModule.trainSet.transform = transTrain

    # ---------- ---------- ---------- ---------- ----------
    cluster_path = dataModule.trainSet.cluster_path
    dataset_name = dataModule.dataset_name
    train_save_path = os.path.join(cluster_path, dataset_name + "_train.pkl")
    # --- RAM load ---
    if export:
        RAM_load(dataModule.trainSet)
        RAM_export(train_save_path, dataModule.trainSet.RAM_data)
    # --- load exported data method ---
    if cluster_import:
        dataModule.trainSet.RAM_data = pkl_import(train_save_path)
        dataModule.trainSet.RAM_load_flag = True
        print("Loaded datasets from ./data/_cluster_.")
    # ---------- ---------- ---------- ---------- ----------

    split = [9/10, 1/10]
    # => We loose properties of the original object when using random_split
    fields_names = dataModule.trainSet.fields_names
    fields_len = dataModule.trainSet.fields_len
    input_len = dataModule.trainSet.input_len
    dataModule.trainSet, dataModule.validSet = random_split(dataModule.trainSet, split, generator=dataModule.seed)
    #
    dataModule.trainSet.dataset_name = dataModule.dataset_name
    dataModule.trainSet.fields_names = fields_names
    dataModule.trainSet.fields_len = fields_len
    dataModule.trainSet.input_len = input_len
    # ---------- ---------- ---------- ---------- ----------


def JHTDB_test(dataModule):

    dataModule.testSet = TurbulenceDataset("Test Extrapolate Mach 0.50-0.52", dataDirs=["data/_pipeline_/JHTDB"],
                                           filterTop=["128_tra"],
                                           filterSim=[(0, 3)], filterFrame=[(500, 750)], sequenceLength=[[65, 2]],  # 60
                                           simFields=["dens", "pres"], simParams=["mach"], printLevel="sim")

    dataParamsKwargs = dict(augmentations=["normalize"], randSeqOffset=True, dimension=2)
    normaKwargs = dict(simFields=["dens", "pres"], simParams=["mach"], normalizeMode="machMixed")
    p_d = DataParams(**dataParamsKwargs, **normaKwargs, dataSize=[128, 64])

    transTrain = Transforms(p_d)
    dataModule.testSet.transform = transTrain


class DataLightningModule(pl.LightningDataModule):
    """
    Custom pytorch_lightning.LightningDataModule to be passed in the Trainer.
    1st step: data is load via the dataset.py custom dataset.
    2nd step: .init()/.load are called. Usually, all the custom setup happens (normalization/proper item fetching).
    3rd step: call the train/valid splitting if necessary.

    Main takeaways:
       Cfg: work directory is fetched from here so globalCfg must be loaded properly.
       seed: fetched from Cfg, must be passed throughout.
       -> dataset_name comes from Cfg.globalCfg.
    """

    def __init__(self, Cfg, pin_memory=True):
        super().__init__()
        self.seed = torch.Generator()
        self.seed.manual_seed(Cfg.globalCfg.seed)
        self.pin_memory = pin_memory

        # --- config ---
        self.Cfg = Cfg
        self.dataset_name = Cfg.globalCfg.dataset_name

        # --- datasets ---
        self.trainSet = None
        self.validSet = None
        self.testSet = None

        self.theWell = ["turbulent_radiative_layer_2D", "MHD_64"]
        self.JHTDB = ["JHTDB"]

    def prepare_data(self, force_call=False, export=None) -> None:
        """
        Prepares data by loading into memory and call respective inits functions. Force_call is set to True for loading
        once manually for config and model initialization. Then Pytorch Lightning re-call so nothing happens.
        Export is only set to true for ./_cluster_/ export (.pkl compact save/load with fixed parameters).
        """
        if force_call or export:
            if self.dataset_name in self.theWell: TheWell_datasetInit(self, export, self.Cfg.cluster)
            if self.dataset_name in self.JHTDB: JHTDB_datasetInit(self, export, self.Cfg.cluster)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.trainSet is None or self.validSet is None:
                sys.exit(".trainSet or .validSet not found. Please call .prepare_data(force_call=True) first.")
        elif stage == 'test':
            if self.dataset_name in self.theWell: TheWell_test(self)
            if self.dataset_name in self.JHTDB: JHTDB_test(self)

    def train_dataloader(self):

        drop_last = True if self.dataset_name in self.JHTDB else False

        loaderKwargs = dict(batch_size=self.Cfg.trainingCfg.batch_size_train, drop_last=drop_last, generator=self.seed)
        trainLoader = DataLoader(self.trainSet, sampler=RandomSampler(self.trainSet), num_workers=4, pin_memory=self.pin_memory,
                                 **loaderKwargs)
        return trainLoader

    def val_dataloader(self):

        drop_last = True if self.dataset_name in self.JHTDB else False

        loaderKwargs = dict(batch_size=self.Cfg.trainingCfg.batch_size_valid, drop_last=drop_last, generator=self.seed)
        # Disable shuffle for validation consistency
        validLoader = DataLoader(self.validSet, sampler=RandomSampler(self.validSet), num_workers=4, pin_memory=self.pin_memory,
                                 shuffle=False, **loaderKwargs)
        return validLoader
