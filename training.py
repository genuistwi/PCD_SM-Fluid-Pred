import torch

from utils.general import warnings_ignore
from utils.config_parser import *

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

from data.module import DataLightningModule
from models.module import ModelLightningModule

from models.callbacks import SaveCfgCallback, WandBLogger, PrintLossCallback
from models.EMA import EMACallback



# --- Preparation ---
warnings_ignore()  # Some should be addressed
torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('medium')


# --- Config loading ---
(trainCfg := ConfigParser(work_dir="")).load("./config/config_list.txt")
trainCfg.WandB = False  # Keep to False
trainCfg.cluster = True

trainCfg.start()  # unique ID attribution. WARNING: do not redo if loading from checkpoint, messes loggers

# --- Paths and directories ---
storage_dir = os.path.join(trainCfg.work_dir, "storage")


# --- Callbacks ---
WandBLoggerCb = WandBLogger(Cfg=trainCfg)  # WandB logs and sweep logs
saveCfgCb = SaveCfgCallback()  # Saves config on first batch
printLossCb = PrintLossCallback()  # Printings while training
EarlyStoppingCb = EarlyStopping(monitor="val_loss", patience=trainCfg.trainingCfg.early_stopping_patience, verbose=True, mode="min")  # Early stopping
ckptCb = ModelCheckpoint(trainCfg.ckpt_dir, monitor="val_loss", filename="checkpoint", save_top_k=1, mode="min")
EMAvgCb = EMACallback(decay=trainCfg.trainingCfg.EMA_rate)  # EMA

callbacks = [*([WandBLoggerCb] if trainCfg.WandB else []), printLossCb, saveCfgCb, ckptCb, EarlyStoppingCb]


# --- Data module ---
dataModule = DataLightningModule(trainCfg, pin_memory=False)  # Fancy dataloader for PyTorch Lightning
dataModule.prepare_data(force_call=True, export=False)  # Dataloaders set as attributes, memory loaded


model_module = ModelLightningModule(trainCfg, dataModule.trainSet)

# --- Trainer ---
trainer = Trainer(
    default_root_dir=storage_dir,                         # Lightning logs directory
    max_epochs=trainCfg.trainingCfg.epochs,
    accelerator="auto",                                    # Can be 'gpu', 'cpu', 'tpu', etc.
    devices="auto",                                        # Tries to find the best setup
    precision=trainCfg.trainingCfg.precision,                                  # Balance
    callbacks=callbacks,
    gradient_clip_val=trainCfg.trainingCfg.grad_clip_val,
)


if __name__ == "__main__": trainer.fit(model_module, datamodule=dataModule)

