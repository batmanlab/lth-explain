import argparse
import os
import sys

import yaml

import lth_pruning.lth_train_completeness as lth_train
import utils

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))


def run_cub(args):
    main_dir = args.main_dir
    # run config
    with open(os.path.join(main_dir, args.config)) as config_file:
        config = yaml.safe_load(config_file)
        _device = utils.get_device()
        print(f"Device: {_device}")
        _seed = config["seed"]
        _data_root = config["data_root"]
        _json_root = config["json_root"]
        _model_arch = config["model_arch"]
        _dataset_name = config["dataset_name"]
        _pretrained = config["pretrained"]
        _transfer_learning = config["transfer_learning"]
        _num_classes = config["num_classes"]
        _logs = config["logs"]
        _bb_layers = config["bb_layers_for_concepts"]
        _concept_names = config["concept_names"]
        _img_size = config["img_size"]
        _batch_size = config["batch_size"]
        _epochs = config["g_epoch"]
        _num_workers = 4

        # 0-Even 1-Odd
        _class_list = config["labels"]
        _num_labels = len(_class_list)

        _g_lr = config["g_lr"]
        _hidden_features = config["hidden_features"]
        _th = config["th"]
        _val_after_th = config["val_after_th"]
        _cav_flattening_type = config["cav_flattening_type"]

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _prune_percent = config["prune_percent"]
        _start_iter = config["start_iter"]
        _end_iter = config["end_iter"]
        _attribute_file_name = config["attribute_file_name"]
        lth_train.train_G_completeness_w_pruning(
            _seed,
            _data_root,
            _json_root,
            _model_arch,
            _num_classes,
            _pretrained,
            _transfer_learning,
            _logs,
            _cav_flattening_type,
            _dataset_name,
            _img_size,
            _start_iter,
            _prune_iterations,
            _prune_type,
            _bb_layers,
            _num_labels,
            _g_lr,
            _batch_size,
            _epochs,
            _hidden_features,
            _th,
            _val_after_th,
            _attribute_file_name,
            _device
        )


def run_derma(args):
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
        _bb_dir = config["bb_dir"]

        _class_list = config["labels"]
        _num_labels = len(_class_list)

        _g_lr = config["g_lr"]
        _hidden_features = config["hidden_features"]
        _th = config["th"]
        _val_after_th = config["val_after_th"]
        _cav_flattening_type = config["cav_flattening_type"]

        _prune_type = config["prune_type"]
        _prune_iterations = config["prune_iterations"]
        _prune_percent = config["prune_percent"]
        _start_iter = config["start_iter"]
        _end_iter = config["end_iter"]
        _attribute_file_name = config["attribute_file_name"]
        _batch_size = config["batch_size"]
        _epochs = config["g_epoch"]
        _class_to_idx = {"benign": 0, "malignant": 1}

        lth_train.train_G_completeness_w_pruning_derma(
            _seed,
            _class_to_idx,
            _bb_dir,
            _data_root,
            _json_root,
            _model_arch,
            _num_classes,
            _pretrained,
            _transfer_learning,
            _logs,
            _cav_flattening_type,
            _dataset_name,
            _img_size,
            _start_iter,
            _prune_iterations,
            _prune_type,
            _bb_layers,
            _num_labels,
            _g_lr,
            _batch_size,
            _epochs,
            _hidden_features,
            _th,
            _val_after_th,
            _attribute_file_name,
            _device
        )


if __name__ == '__main__':
    print("Concept completeness g training: ")
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-c", default="HAM10k")
    parser.add_argument(
        "--main_dir", "-m", default="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/")
    args = parser.parse_args()
    if args.dataset == "cub":
        args.config = "config/BB_cub.yaml"
        run_cub(args)
    elif args.dataset == "HAM10k":
        args.config = "config/BB_derma.yaml"
        run_derma(args)
