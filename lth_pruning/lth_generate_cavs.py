import os
import pickle
import random

import numpy as np
import torch

import concept_activations.cav_generation as cavs
import utils
from concept_activations.cavs_for_derma import train_for_concepts


def generate_cavs_with_Pruning(
        seed,
        prune_type,
        dataset_name,
        start_iter,
        prune_iterations,
        logs,
        model_arch,
        bb_layers,
        concept_names,
        cav_flattening_type
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for layer in bb_layers:
        activations_path = os.path.join(
            logs, "activations", "Pruning", model_arch, dataset_name, "BB_act", f"Prune_type_{prune_type}", layer
        )

        cav_path = os.path.join(
            logs, "activations", "Pruning", model_arch, dataset_name, "cavs", f"Prune_type_{prune_type}",
            f"cav_flattening_type_{cav_flattening_type}", layer
        )

        utils.create_dir(
            path_dict={
                "path_name": cav_path,
                "path_type": "cavs-of-BB"
            })

        for _ite in range(start_iter, prune_iterations):
            print(f"Prune iteration: {_ite} =======================================>")
            train_activations_file = f"train_activations_prune_iteration_{_ite}.h5"
            val_activation_file = f"val_activations_prune_iteration_{_ite}.h5"
            test_activation_file = f"test_activations_prune_iteration_{_ite}.h5"

            train_GT_file = f"train_np_attr_GT_prune_iteration_{_ite}.npy"
            val_GT_file = f"val_np_attr_GT_prune_iteration_{_ite}.npy"
            test_GT_file = f"test_np_attr_GT_prune_iteration_{_ite}.npy"

            if dataset_name == "mnist":
                train_cavs, train_cav_cls_report = cavs.generate_cavs_using_pruning(
                    concept_names,
                    cav_flattening_type,
                    bb_layers,
                    activations_path,
                    train_activations_file,
                    val_activation_file,
                    # test_activation_file,
                    train_GT_file,
                    val_GT_file,
                    # test_GT_file
                )
            elif dataset_name == "cub":
                train_cavs, train_cav_cls_report, train_models = cavs.generate_cavs_using_pruning(
                    concept_names,
                    cav_flattening_type,
                    bb_layers,
                    activations_path,
                    train_activations_file,
                    val_activation_file,
                    # test_activation_file,
                    train_GT_file,
                    val_GT_file,
                    # test_GT_file,
                    multi_label=True
                )

            if cav_flattening_type == "max_pooled":
                cav_file_name = f"max_pooled_train_cavs_prune_iteration_{_ite}.pkl"
                cls_report_file = f"max_pooled_train_cls_report_prune_iteration_{_ite}.pkl"
                cav_model_name = f"max_pooled_train_model_prune_iteration_{_ite}.pth.tar"
            elif cav_flattening_type == f"flattened":
                cav_file_name = f"flattened_train_cavs_prune_iteration_{_ite}.pkl"
                cls_report_file = f"flattened_train_cls_report_prune_iteration_{_ite}.pkl"
                cav_model_name = f"flattened_train_model_prune_iteration_{_ite}.pth.tar"
            elif cav_flattening_type == f"adaptive_avg_pooled":
                cav_file_name = f"adaptive_avg_pooled_train_cavs_prune_iteration_{_ite}.pkl"
                cls_report_file = f"adaptive_avg_pooled_train_cls_report_prune_iteration_{_ite}.pkl"
                cav_model_name = f"adaptive_avg_pooled_train_model_prune_iteration_{_ite}.pth.tar"
            else:
                cav_file_name = f"avg_pooled_train_cavs_prune_iteration_{_ite}.pkl"
                cls_report_file = f"avg_pooled_train_cls_report_prune_iteration_{_ite}.pkl"
                cav_model_name = f"avg_pooled_train_model_prune_iteration_{_ite}.pth.tar"

            cav_file = open(os.path.join(cav_path, cav_file_name), "wb")
            pickle.dump(train_cavs, cav_file)
            cav_file.close()
            for layer in bb_layers:
                torch.save(
                    train_models[layer].state_dict(),
                    os.path.join(cav_path, cav_model_name)
                )

            concept_classifier_report_file = open(
                os.path.join(cav_path, cls_report_file),
                "wb"
            )
            pickle.dump(train_cav_cls_report, concept_classifier_report_file)
            print(f"Activation dictionary is saved in the location: {cav_path}")


def generate_cavs_with_Pruning_derma(
        seed,
        prune_type,
        dataset_name,
        start_iter,
        prune_iterations,
        logs,
        model_arch,
        derm7_folder,
        derm7_meta,
        C,
        bb_dir,
        model_name,
        derm7_train_idx,
        derm7_val_idx,
        concept_names,
        n_samples,
        cav_flattening_type
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    layer = "derma"
    cav_path = os.path.join(
        logs, "activations", "Pruning", model_arch, dataset_name, "cavs", f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}", layer
    )
    checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
    utils.create_dir(
        path_dict={
            "path_name": cav_path,
            "path_type": "cavs-of-BB"
        })

    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")
        train_for_concepts(
            derm7_folder, cav_path, derm7_meta, C, bb_dir, model_name, derm7_train_idx, derm7_val_idx, n_samples,
            concept_names, _ite, checkpoint_path
        )
        print(f"Activation dictionary is saved in the location: {cav_path}")
