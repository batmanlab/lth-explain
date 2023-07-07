import os
import pickle
import random
import time

import numpy as np
import torch
from torchvision import transforms

import concept_activations.TCAV as TCAV
import concept_activations.concept_activations_utils as ca_utils
import lth_pruning.pruning_utils as pruning_utils
import utils
from concept_activations.flatten_LR import Flatten_LR
from dataset.dataset_derma import load_HAM10k_data
from model_factory.model_meta import Model_Meta


def cal_TCAV_w_pruning(
        seed,
        data_root,
        json_root,
        model_arch,
        num_classes,
        pretrained,
        transfer_learning,
        logs,
        cav_flattening_type,
        dataset_name,
        img_size,
        start_iter,
        prune_iterations,
        prune_type,
        bb_layer,
        batch_size,
        concept_names,
        class_list,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform_params = {
        "img_size": img_size
    }
    start = time.time()
    test_loader = pruning_utils.get_test_dataloader(
        dataset_name,
        data_root,
        json_root,
        batch_size,
        transform_params
    )
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")
    bb_checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
    cav_path = os.path.join(
        logs,
        "activations",
        "Pruning",
        model_arch,
        dataset_name,
        "cavs",
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}"
    )

    activations_path = os.path.join(
        logs,
        "activations",
        "Pruning",
        model_arch,
        dataset_name,
        "BB_act",
        f"Prune_type_{prune_type}"
    )

    prune_stat_path = os.path.join(
        logs,
        "predictions",
        "prune-statistics",
        model_arch,
        dataset_name,
        f"Prune_type_{prune_type}",
        f"cav_flattening_type_{cav_flattening_type}"
        "TCAV-scores"
    )
    utils.create_dir(
        path_dict={
            "path_name": prune_stat_path,
            "path_type": "prune_stat_path-for-each-prune-iteration"
        })

    for _ite in range(start_iter, prune_iterations):
        print(f"Prune iteration: {_ite} =======================================>")
        test_activations_file = f"test_activations_prune_iteration_{_ite}.h5"
        concept_vectors = ca_utils.get_concept_vectors_for_pruning(
            _ite,
            cav_path,
            bb_layer,
            cav_flattening_type
        )

        bb_model = pruning_utils.load_BB_model_w_pruning(
            model_arch,
            num_classes,
            pretrained,
            transfer_learning,
            dataset_name,
            device,
            _ite,
            bb_checkpoint_path
        )
        bb_model.eval()
        if type(bb_layer) == str:
            bb_model_meta = Model_Meta(bb_model, [bb_layer])
        else:
            bb_model_meta = Model_Meta(bb_model, bb_layer)

        start = time.time()

        if len(class_list) == 2:
            stat_dict = TCAV.calculate_cavs_binary_classification(
                test_loader,
                concept_vectors,
                bb_model,
                bb_model_meta,
                cav_flattening_type,
                bb_layer,
                concept_names,
                class_list,
                model_arch
            )
            done = time.time()
            elapsed = done - start
            print("Time to complete this iteration: " + str(elapsed) + " secs")
            metric_file_per_iter = open(os.path.join(
                prune_stat_path,
                f"TCAV_scores_file_pruning_iter_{_ite}.pkl"
            ), "wb")
            pickle.dump(stat_dict, metric_file_per_iter)
            metric_file_per_iter.close()


def cal_TCAV_w_pruning_multiclass(
        seed,
        data_root,
        json_root,
        model_arch,
        num_classes,
        pretrained,
        transfer_learning,
        logs,
        cav_flattening_type,
        dataset_name,
        img_size,
        start_iter,
        prune_iterations,
        prune_type,
        bb_layer,
        batch_size,
        concept_names,
        tcav_to_predict,
        class_labels,
        attribute_file_name,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    layer = bb_layer
    start = time.time()
    test_loader = None
    if dataset_name == "cub":
        test_loader = pruning_utils.get_test_dataloader_cub(
            dataset_name,
            data_root,
            json_root,
            batch_size,
            img_size,
            attribute_file_name
        )
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")

    for class_label in tcav_to_predict:
        concepts = tcav_to_predict[class_label]
        print(class_label, '====>>>', concepts)
        labels_index_TCAV = class_labels.index(class_label)
        concept_index_list_TCAV = pruning_utils.get_class_index_for_TCAV(concepts, concept_names)

        bb_checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
        cav_path = os.path.join(
            logs,
            "activations",
            "Pruning",
            model_arch,
            dataset_name,
            "cavs",
            f"Prune_type_{prune_type}",
            f"cav_flattening_type_{cav_flattening_type}",
            layer
        )

        activations_path = os.path.join(
            logs,
            "activations",
            "Pruning",
            model_arch,
            dataset_name,
            "BB_act",
            f"Prune_type_{prune_type}",
            layer
        )

        prune_stat_path = os.path.join(
            logs,
            "predictions",
            "prune-statistics",
            model_arch,
            dataset_name,
            f"Prune_type_{prune_type}",
            f"cav_flattening_type_{cav_flattening_type}",
            class_label
        )
        utils.create_dir(
            path_dict={
                "path_name": prune_stat_path,
                "path_type": "prune_stat_path-for-each-prune-iteration"
            })

        for _ite in range(start_iter, prune_iterations):
            print(f"*********************************************** "
                  f"Prune iteration: {_ite} "
                  f"***********************************************")
            t = None
            if cav_flattening_type == "adaptive_avg_pooled":
                t = Flatten_LR(ip_size=2048, op_size=108).to(device)
            elif cav_flattening_type == "flattened":
                t = Flatten_LR(ip_size=2048 * 14 * 14, op_size=108).to(device)
            checkpoint_t = os.path.join(cav_path, f"{cav_flattening_type}_train_model_prune_iteration_{_ite}.pth.tar")
            t.load_state_dict(torch.load(checkpoint_t))
            t.eval()
            torch_concept_vector = t.model[0].weight.cpu().detach()
            bb_model = pruning_utils.load_BB_model_w_pruning(
                model_arch,
                num_classes,
                pretrained,
                transfer_learning,
                dataset_name,
                device,
                _ite,
                bb_checkpoint_path
            )
            bb_model.eval()
            bb_model_meta = Model_Meta(bb_model, [layer])
            start = time.time()

            stat_dict = TCAV.calculate_cavs_multiclass(
                test_loader,
                torch_concept_vector,
                bb_model,
                bb_model_meta,
                cav_flattening_type,
                bb_layer,
                model_arch,
                class_labels,
                labels_index_TCAV,
                concept_names,
                concept_index_list_TCAV
            )
            done = time.time()
            elapsed = done - start
            print("Time to complete this iteration: " + str(elapsed) + " secs")
            print(f"TCAV Stats for this iteration: {_ite} ")
            print(stat_dict)
            metric_file_per_iter = open(os.path.join(
                prune_stat_path,
                f"TCAV_scores_file_pruning_iter_{_ite}.pkl"
            ), "wb")
            pickle.dump(stat_dict, metric_file_per_iter)
            metric_file_per_iter.close()


def cal_TCAV_w_pruning_multiclass_derma(
        seed,
        class_to_idx,
        bb_dir,
        data_root,
        json_root,
        model_arch,
        num_classes,
        pretrained,
        transfer_learning,
        logs,
        cav_flattening_type,
        dataset_name,
        img_size,
        start_iter,
        prune_iterations,
        prune_type,
        bb_layer,
        batch_size,
        concept_names,
        tcav_to_predict,
        class_labels,
        attribute_file_name,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    layer = bb_layer
    start = time.time()
    test_loader = None
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize
        ]
    )
    _, test_loader, idx_to_class = load_HAM10k_data(
        seed, data_root, transform, class_to_idx, batch_size, mode="train"
    )
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")

    for class_label in tcav_to_predict:
        concepts = tcav_to_predict[class_label]
        print(class_label, '====>>>', concepts)
        labels_index_TCAV = class_labels.index(class_label)
        concept_index_list_TCAV = pruning_utils.get_class_index_for_TCAV(concepts, concept_names)

        bb_checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
        cav_path = os.path.join(
            logs,
            "activations",
            "Pruning",
            model_arch,
            dataset_name,
            "cavs",
            f"Prune_type_{prune_type}",
            f"cav_flattening_type_{cav_flattening_type}",
            layer
        )

        prune_stat_path = os.path.join(
            logs,
            "predictions",
            "prune-statistics",
            model_arch,
            dataset_name,
            f"Prune_type_{prune_type}",
            f"cav_flattening_type_{cav_flattening_type}",
            class_label
        )
        utils.create_dir(
            path_dict={
                "path_name": prune_stat_path,
                "path_type": "prune_stat_path-for-each-prune-iteration"
            })

        for _ite in range(start_iter, prune_iterations):
            print(
                f"************************************************** "
                f"Prune iteration: {_ite} "
                f"**************************************************"
            )
            concepts_dict = pickle.load(
                open(os.path.join(cav_path, f"derma_ham10000_0.01_50_ite_{_ite}.pkl"), "rb")
            )
            cavs = []
            for key in concepts_dict.keys():
                cavs.append(concepts_dict[key][0][0].tolist())
            cavs = np.array(cavs)
            torch_concept_vector = torch.from_numpy(cavs).to(device, dtype=torch.float32)

            model, model_bottom, model_top = pruning_utils.load_BB_model_w_pruning_derma(
                bb_dir,
                device,
                _ite,
                bb_checkpoint_path
            )
            model.eval()
            start = time.time()

            stat_dict = TCAV.calculate_cavs_multiclass_derma(
                test_loader,
                torch_concept_vector,
                model,
                model_bottom,
                model_top,
                cav_flattening_type,
                bb_layer,
                model_arch,
                class_labels,
                labels_index_TCAV,
                concept_names,
                concept_index_list_TCAV
            )
            done = time.time()
            elapsed = done - start
            print("Time to complete this iteration: " + str(elapsed) + " secs")
            print(f"TCAV Stats for this iteration: {_ite} ")
            print(stat_dict)
            metric_file_per_iter = open(os.path.join(
                prune_stat_path,
                f"TCAV_scores_file_pruning_iter_{_ite}.pkl"
            ), "wb")
            pickle.dump(stat_dict, metric_file_per_iter)
            metric_file_per_iter.close()
