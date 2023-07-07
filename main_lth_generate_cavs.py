import argparse
import os
import sys

import yaml

import utils
from lth_pruning import lth_generate_cavs

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))


def run_mnist(args):
    print("Generate CAVs of BB using Pruning by LTH for MNIST")
    main_dir = args.main_dir

    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _bb_layers = config["bb_layers_for_concepts"]
        _img_size = config["img_size"]
        _seed = config["seed"]
        _dataset_name = config["dataset_name"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]

        _lr = config["lr"]
        _logs = config["logs"]
        _num_classes = config["num_classes"]
        _epochs = config["epochs"]
        _device = utils.get_device()

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _prune_percent = config["prune_percent"]
        _start_iter = config["start_iter"]
        _end_iter = config["end_iter"]
        _resample = config["resample"]
        _epsilon = config["epsilon"]
        _concept_names = config["concept_names"]
        _cav_flattening_type = config["cav_flattening_type"]

        lth_generate_cavs.generate_cavs_with_Pruning(
            _seed,
            _prune_type,
            _dataset_name,
            _start_iter,
            _prune_iterations,
            _logs,
            _model_arch,
            _bb_layers,
            _concept_names,
            _cav_flattening_type
        )


def run_cub(args):
    print("Generate CAVs of BB using Pruning by LTH for CUB")
    main_dir = args.main_dir

    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _bb_layers = config["bb_layers_for_concepts"]
        _img_size = config["img_size"]
        _seed = config["seed"]
        _dataset_name = config["dataset_name"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]

        _lr = config["lr"]
        _logs = config["logs"]
        _num_classes = config["num_classes"]
        _device = utils.get_device()

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _prune_percent = config["prune_percent"]
        _start_iter = config["start_iter"]
        _end_iter = config["end_iter"]
        _resample = config["resample"]
        _epsilon = config["epsilon"]
        _concept_names = config["concept_names"]
        _cav_flattening_type = config["cav_flattening_type"]

        lth_generate_cavs.generate_cavs_with_Pruning(
            _seed,
            _prune_type,
            _dataset_name,
            _start_iter,
            _prune_iterations,
            _logs,
            _model_arch,
            _bb_layers,
            _concept_names,
            _cav_flattening_type
        )


def run_derma(args):
    print("Generate CAVs of BB using Pruning by LTH for HAM10k")
    # parse arguments
    main_dir = args.main_dir

    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _bb_layers = config["bb_layers_for_concepts"]
        _img_size = config["img_size"]
        _seed = config["seed"]
        _dataset_name = config["dataset_name"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]

        _lr = config["lr"]
        _logs = config["logs"]
        _num_classes = config["num_classes"]
        _device = utils.get_device()

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _prune_percent = config["prune_percent"]
        _start_iter = config["start_iter"]
        _end_iter = config["end_iter"]
        _resample = config["resample"]
        _epsilon = config["epsilon"]
        _concept_names = config["concept_names"]
        _cav_flattening_type = config["cav_flattening_type"]
        _bb_dir = config["bb_dir"]
        _derm7_folder = config["derm7_folder"]
        _derm7_meta = config["derm7_meta"]
        _C = config["C"]
        _model_name = config["model_name"]
        _derm7_train_idx = config["derm7_train_idx"]
        _derm7_val_idx = config["derm7_val_idx"]
        _n_samples = config["n_samples"]

        lth_generate_cavs.generate_cavs_with_Pruning_derma(
            _seed,
            _prune_type,
            _dataset_name,
            _start_iter,
            _prune_iterations,
            _logs,
            _model_arch,
            _derm7_folder,
            _derm7_meta,
            _C,
            _bb_dir,
            _model_name,
            _derm7_train_idx,
            _derm7_val_idx,
            _concept_names,
            _n_samples,
            _cav_flattening_type
        )


if __name__ == '__main__':
    print("Train BB using pruning by LTH")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-c", default="mnist")
    parser.add_argument(
        "--main_dir", "-m", default="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/")
    args = parser.parse_args()
    if args.dataset == "cub":
        args.config = "config/BB_cub.yaml"
        run_cub(args)
    elif args.dataset == "mnist":
        args.config = "config/BB_mnist.yaml"
        run_mnist(args)
    elif args.dataset == "HAM10k":
        args.config = "config/BB_derma.yaml"
        run_derma(args)
