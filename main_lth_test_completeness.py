import argparse
import os
import sys

import yaml

import lth_pruning.lth_test_completeness as lth_train
import utils

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))

if __name__ == '__main__':
    print("Concept completeness g training: ")
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config/BB_cub.yaml")
    parser.add_argument(
        "--main_dir", "-m", default="/ocean/projects/asc170022p/shg121/PhD/Project_Pruning/")
    args = parser.parse_args()
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
        lth_train.test_pruned_model_with_completeness_score(
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
            _batch_size,
            _epochs,
            _g_lr,
            _hidden_features,
            _th,
            _val_after_th,
            _attribute_file_name,
            _device
        )
