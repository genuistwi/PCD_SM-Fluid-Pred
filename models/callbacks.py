import pytorch_lightning as pl
import os
import wandb
from pytorch_lightning.utilities import rank_zero_only
from ray import tune
import dill
import pickle


class SyncCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    def on_train_start(self, trainer, pl_module):
        trainer.strategy.barrier()

    def on_train_end(self, trainer, pl_module):
        trainer.strategy.barrier()


class SaveCfgCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.saved = False  # Track if Cfg has already been saved

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        """
        Check if a checkpoint has been saved and save Cfg only once.
        """
        if not self.saved and trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            save_dir = pl_module.Cfg.ckpt_dir
            cfg_save_path = os.path.join(save_dir, "Cfg.pkl")
            if hasattr(pl_module, "Cfg"):
                with open(cfg_save_path, "wb") as f:
                    dill.dump(pl_module.Cfg, f)
                print(f"Saved model Cfg to {cfg_save_path}")
                self.saved = True  # Ensure it's saved only once

class PrintLossCallback(pl.Callback):
    """
    Custom callback that prints:
    - The current step loss at the end of each batch
    - The average epoch loss at the end of each epoch
    """
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # Look up the step-level train loss in trainer.callback_metrics.
        # By default, if you do self.log("train_loss", ..., on_step=True, on_epoch=True),
        # Lightning will create train_loss_step and train_loss_epoch under callback_metrics.
        step_loss = trainer.callback_metrics.get("train_loss_step")
        if step_loss is not None and not trainer.sanity_checking:
            print(f"Step {trainer.global_step} - train_loss_step: {step_loss:.4f}")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Look up the epoch-level train loss
        epoch_loss = trainer.callback_metrics.get("train_loss_epoch")
        if epoch_loss is not None and not trainer.sanity_checking:
            print(f"Epoch {trainer.current_epoch} - train_loss_epoch: {epoch_loss:.4f}")

        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        print(f"Epoch {trainer.current_epoch}: lr={current_lr}")

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None and not trainer.sanity_checking:
            print(f"Epoch {trainer.current_epoch} - val_loss: {val_loss:.4f}")


class WandBLogger(pl.Callback):
    def __init__(self, Cfg):
        self.Cfg = Cfg

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):

        wandb.login(key=self.Cfg.globalCfg.WandB_key)
        wandb.init(project=self.Cfg.project_name, reinit=True)


        modelCfg = pl_module.Cfg.modelCfg
        globalCfg = pl_module.Cfg.globalCfg
        sdeCfg = pl_module.Cfg.sdeCfg
        trainingCfg = pl_module.Cfg.trainingCfg

        project_name = pl_module.Cfg.project_name

        print("-> WandB tracking initialized, project name: ", project_name)

        config = {
            "ID": pl_module.Cfg.ID,

            # --- model config ---
            "Model name": modelCfg.model_name,
            "Conditional": modelCfg.conditional,

            # --- global config ---
            "Seed": globalCfg.seed,
            "Dataset name": globalCfg.dataset_name,

            # --- SDE config ---
            "SDE name": sdeCfg.sde_name,
            "Num scales (N)": sdeCfg.num_scales,
            "T_max (T)": sdeCfg.T_max,
            "Sigma min": sdeCfg.sigma_min,
            "Sigma max": sdeCfg.sigma_max,
            "Beta min": sdeCfg.beta_min,
            "Beta max": sdeCfg.beta_max,

            # --- training config ---
            "Max epochs": trainingCfg.epochs,
            "Learning rate": trainingCfg.lr,
            "Batch size train": trainingCfg.batch_size_train,
            "Batch size valid": trainingCfg.batch_size_valid,
            "Likelihood weighting": trainingCfg.likelihood_weighting,
            "Energy loss": trainingCfg.energy_loss,
            "Lambda energy loss": trainingCfg.e_lambda,
        }

        wandb.config.update(config, allow_val_change=True)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        train_loss = trainer.callback_metrics["train_loss"]
        train_eLoss = trainer.callback_metrics["train_eLoss"]
        log = {
            "step": trainer.global_step,
            "batch idx": batch_idx,
            # "epoch": trainer.current_epoch,
            "train loss": train_loss,
            "train eLoss": train_eLoss,
        }
        if not trainer.sanity_checking: wandb.log(log)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics["train_loss"]
        train_eLoss = trainer.callback_metrics["train_eLoss"]
        log = {
            "epoch": trainer.current_epoch,
            "train loss": train_loss,
            "train eLoss": train_eLoss,
        }
        if not trainer.sanity_checking: wandb.log(log)


    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics["val_loss"]
        MSE_ref = trainer.callback_metrics["MSE_ref"]
        MSE_est = trainer.callback_metrics["MSE_est"]
        log = {
            "epoch": trainer.current_epoch,
            "val_loss": val_loss,
            "MSE_ref": MSE_ref,
            "MSE_est": MSE_est,
        }
        if not trainer.sanity_checking: wandb.log(log)


    def on_train_end(self, trainer, pl_module):
        wandb.finish()
