""" Training configuration file. """
from utils.objects.utils import EmptyObj

exec(open("config/global.py").read(), namespace := {})
optimCfg = EmptyObj()


if namespace["dataset_name"] == "MHD_64":
    # --- MHD 64/32 ---
    epochs = 350
    lr = .0001
    batch_size_train = 8
    batch_size_valid = 2
    early_stopping_patience = 20  # Number of epochs with no improvement after which early stopping
    scheduler_patience = 6  # Number of epochs with no improvement after which LR will be reduced
    scheduler_factor = .5
    find_unused_parameters = False
    gradient_as_bucket_view = False
    precision = "16-mixed"

    grad_clip_val = 0.4

elif namespace["dataset_name"] == "JHTDB":
    epochs = 600
    lr = .0003
    batch_size_train = 164
    batch_size_valid = 100
    early_stopping_patience = 90
    scheduler_patience = 25
    scheduler_factor = .6
    find_unused_parameters = True
    gradient_as_bucket_view = True
    precision = "32"

    grad_clip_val = 1.

else:
    epochs = 700
    lr = .0003
    batch_size_train = 128
    batch_size_valid = 128
    early_stopping_patience = 140
    scheduler_patience = 30
    scheduler_factor = .5
    find_unused_parameters = True
    gradient_as_bucket_view = True
    precision = "32"

    grad_clip_val = 1.


EMA_rate      = 0.999

optimCfg.weight_decay   = 0.0
optimCfg.optimizer_name = "Adam"
optimCfg.beta1          = 0.9
optimCfg.eps            = 1e-8

optimCfg.warmup = 1000
ema_update_every = 10  # batches

# --- loss parameters ---
likelihood_weighting = False
energy_loss          = False
e_lambda             = .35
