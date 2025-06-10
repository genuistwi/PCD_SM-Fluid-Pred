from copy import deepcopy
from typing import Optional, Union, Dict, Any

import pytorch_lightning as pl
import torch
from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only


class EMACallback(pl.Callback):
    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in
                                       self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

    @overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_save_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ):
        return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    def on_load_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
    ):
        self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
        self.ema_state_dict = callback_state["ema_state_dict"]
