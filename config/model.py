exec(open("config/global.py").read(), namespace := {})


if namespace["dataset_name"] == "MHD_64":
    model_name = "Custom_CondAttUnet_lowSkip_3D"
else:
    model_name = "Custom_CondAttUnet_lowSkip"

conditional = True
