import argparse
import os
import sys

import yaml

import lth_pruning.lth_pcbm_train_eval as lth_train
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
        _epochs = config["pcbm_epoch"]
        _num_workers = 4

        # 0-Even 1-Odd
        _class_list = config["labels"]
        _num_labels = len(_class_list)

        _pcbm_lr = config["pcbm_lr"]
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
        _pcbm_alpha = config["pcbm_alpha"]
        _pcbm_l1_ratio = config["pcbm_l1_ratio"]
        lth_train.train_eval_pcbm_w_pruning(
            _seed,
            _model_arch,
            _num_classes,
            _pretrained,
            _transfer_learning,
            _logs,
            _cav_flattening_type,
            _dataset_name,
            _start_iter,
            _prune_iterations,
            _prune_type,
            _bb_layers,
            _num_labels,
            _pcbm_lr,
            _pcbm_alpha,
            _pcbm_l1_ratio,
            _epochs,
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
        _pcbm_lr = config["pcbm_lr"]
        _pcbm_alpha = config["pcbm_alpha"]
        _pcbm_l1_ratio = config["pcbm_l1_ratio"]
        _epochs = config["pcbm_epoch"]
        _batch_size = config["batch_size"]
        _class_to_idx = {"benign": 0, "malignant": 1}

        lth_train.train_eval_pcbm_w_pruning_derma(
            _seed,
            _bb_dir,
            _data_root,
            _class_to_idx,
            _batch_size,
            _model_arch,
            _num_classes,
            _pretrained,
            _transfer_learning,
            _logs,
            _cav_flattening_type,
            _dataset_name,
            _start_iter,
            _prune_iterations,
            _prune_type,
            _bb_layers,
            _num_labels,
            _pcbm_lr,
            _pcbm_alpha,
            _pcbm_l1_ratio,
            _epochs,
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
