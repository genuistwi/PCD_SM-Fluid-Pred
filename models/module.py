from pytorch_lightning import LightningModule
from models.losses import *


from models.utils import *
from SDE.utils import *


class ModelLightningModule(LightningModule):
    def __init__(self, Cfg, data_set):
        super().__init__()

        self.Cfg = Cfg

        self.model = get_model(self.Cfg, data_set)
        self.SDE = get_sde(self.Cfg)
        self.steps_per_epoch = len(data_set)

        self.score_fn = get_score_fn(self.Cfg, self.SDE, self.model)
        self.loss_fn = get_loss_fn(self.SDE, self.score_fn)


        self.energyLoss = get_energyLoss_fn(dim = 3 if data_set.dataset_name == "MHD_64" else 2)

        self.val_loss = get_val_loss_fn(self.SDE, self.score_fn)

        self.lr = Cfg.trainingCfg.lr
        self.step = 0
        self.eLoss = self.Cfg.trainingCfg.energy_loss
        try:
            self.eLambda = self.Cfg.trainingCfg.e_lambda
        except:
            print("Warning: eLambda is not defined, maybe old run loaded and wasn't defined yet.")
            self.eLambda = 1e-2

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def training_step(self, batch, batch_idx):

        X_tau_cond, X_tau = batch

        X_tau = self.model.input_format(X_tau)
        X_tau_cond = self.model.cond_format(X_tau_cond)

        loss, X_hat = self.loss_fn(X_tau, X_tau_cond)
        self.log('train_loss', loss, on_epoch=True, on_step=True, sync_dist=True)

        if self.eLoss:
            u_p_p, u_p_t, u_p_h = self.model.energyLoss_format(X_tau, X_tau_cond, X_hat)
            energyLoss = self.energyLoss(u_p_p, u_p_t, u_p_h)
            self.log('train_eLoss', energyLoss, on_epoch=True, on_step=True, sync_dist=True)
            loss += self.eLambda * energyLoss

        else:
            self.log('train_eLoss', 9.999, on_epoch=True, on_step=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        X_tau_cond, X_tau = batch

        X_tau = self.model.input_format(X_tau)
        X_tau_cond = self.model.cond_format(X_tau_cond)

        val_loss, MSE_ref, MSE_est = self.val_loss(X_tau, X_tau_cond)

        if torch.isnan(MSE_est).any():
            MSE_est = torch.tensor(3., device=X_tau.device)

        self.log('val_loss', val_loss, on_epoch=True, on_step=True, sync_dist=True)
        self.log('MSE_ref', MSE_ref, on_epoch=True, on_step=True, sync_dist=True)
        self.log('MSE_est', MSE_est, on_epoch=True, on_step=True, sync_dist=True)

        return MSE_est


    def configure_optimizers(self):
        """
        Define the optimizer and (optionally) a learning rate scheduler.
        The returned object can be:
          - A single optimizer
          - A list or dict of optimizers, or
          - A dict with entries 'optimizer' and (optionally) 'lr_scheduler'
        """
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # LR_scheduler
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # Look for a decrease in the monitored metric (e.g., val_loss)
            factor=self.Cfg.trainingCfg.scheduler_factor,  # Factor by which the learning rate will be reduced
            patience=self.Cfg.trainingCfg.scheduler_patience,  # Number of epochs with no improvement after which LR will be reduced
            verbose=True  # Print a message when the learning rate is reduced
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,  # The LR scheduler instance
                "monitor": "val_loss",   # Metric to monitor (used with some schedulers)
                "interval": "epoch",     # Step the scheduler every epoch
                # "frequency": 1          # How many epochs between scheduler steps, 1 for plateau, None for OneCycleLR
            }
        }

