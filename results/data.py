import numpy as np
from utils.general import denormalize
from results.utils import load_objects
from einops import rearrange
import copy



def multiple_GT_loading(any_method_string, any_ID, any_version, test_cases, means, stds, dataset_name):

    # ID doesn't matter, GT is the same across the entire dataset's version
    # The same is true for method string

    assert dataset_name in ["JHTDB", "turbulent_radiative_layer_2D", "MHD_64"]
    loadKwargs = dict(storage_path="./storage", dataset_name=dataset_name)
    GTs = dict()
    mask = None

    for test_case_number in test_cases:  # A test case is typically a case out of the 3 of the test dataset

        sample, _ = load_objects(sub_dir_name=any_method_string, sampleCfg_id=any_ID,
                                 test_case_number=test_case_number, version=any_version, **loadKwargs)

        if dataset_name == "JHTDB":
            mask = sample["mask"].detach().cpu().numpy().T
            GT = rearrange(sample["GT"].detach().cpu().numpy(), "T F Lx Ly -> T F Ly Lx")
            GT = denormalize(GT, means, stds)[:60] * mask  # Cylinder obstacle

        elif dataset_name == "turbulent_radiative_layer_2D":
            GT = rearrange(sample["GT"].detach().cpu().numpy(), "T F Lx Ly -> T F Lx Ly")
            GT = denormalize(GT, means, stds)[:60]

        elif dataset_name == "MHD_64":
            GT = rearrange(sample["GT"].detach().cpu().numpy(), "T F Lx Ly Lz-> T F Lx Ly Lz")
            GT = denormalize(GT, means, stds)[:20]

        else:
            raise NotImplementedError

        GTs[test_case_number] = GT

    GTs["mask"] = mask
    fields_name = sample["fields_names"]

    return GTs, fields_name


def make_results_dicts(GTs, regularization, sde_names, test_cases):

    sde_dict = {sde: {} for sde in sde_names}

    test_sets = {}
    for gt, i in zip(GTs, test_cases):
        test_set = {"GT": {"pred": GTs[i]}}
        for reg in regularization:
            test_set[reg] = copy.deepcopy(sde_dict)
        test_sets[i] = test_set

    return test_sets


def load_multiple_results(method_strings, IDs, regularization, sde_names,
                          empty_test_sets, versions, test_cases,
                          means, stds, dataset_name, mask=None):

    assert dataset_name in ["JHTDB", "turbulent_radiative_layer_2D", "MHD_64"]
    loadKwargs = dict(storage_path="./storage", dataset_name=dataset_name)
    test_sets = {}
    for test_case_number in test_cases:
        empty_test_set = empty_test_sets[test_case_number]
        c = 0
        # same order: E_loss_False__ODE_False / "regFalse" && E_loss_True__ODE_False / "regTrue"
        for (method_string, regu) in zip(method_strings, regularization):
            # IDs order should match regu false vp,subvp,ve SDE then regu true vp,subvp,ve SDE
            for sde_name in sde_names:
                ID = IDs[c]
                empty_test_set[regu][sde_name]["pred"] = {}
                for version in versions:
                    sample, _ = load_objects(sub_dir_name=method_string, sampleCfg_id=ID,
                                 test_case_number=test_case_number, version=version, **loadKwargs)
                    if dataset_name == "JHTDB":
                        pred = rearrange(sample["pred"].detach().cpu().numpy(), "T F Lx Ly -> T F Ly Lx")
                        pred = denormalize(pred, means, stds)[:60] * mask
                    elif dataset_name == "turbulent_radiative_layer_2D":
                        pred = rearrange(sample["pred"].detach().cpu().numpy(), "T F Lx Ly -> T F Lx Ly")
                        pred = denormalize(pred, means, stds)[:60]
                    elif dataset_name == "MHD_64":
                        pred = rearrange(sample["pred"].detach().cpu().numpy(), "T F Lx Ly Lz-> T F Lx Ly Lz")
                        pred = denormalize(pred, means, stds)[:20]
                    else:
                        raise NotImplementedError
                    empty_test_set[regu][sde_name]["pred"][version] = pred
                c += 1
        test_sets[test_case_number] = empty_test_set

    return test_sets

