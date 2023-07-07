import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import concept_activations.concept_activations_utils as ca_utils
import concept_activations.concept_completeness_train as cav_train
import lth_pruning.pruning_utils as prun_utils
import utils
from dataset.dataset_derma import load_HAM10k_data
from model_factory.model_meta import Model_Meta
from torchvision import transforms

def train_G_completeness_w_pruning(
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
        bb_layers,
        num_labels,
        g_lr,
        batch_size,
        epochs,
        hidden_features,
        th,
        val_after_th,
        attribute_file_name,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # start = time.time()
    # train_loader, val_loader = ca_utils.get_dataloader_cub(
    #     data_root,
    #     json_root,
    #     dataset_name,
    #     img_size,
    #     batch_size,
    #     attribute_file_name
    # )
    # done = time.time()
    # elapsed = done - start
    # print("Time to load the dataset from disk: " + str(elapsed) + " secs")

    bb_checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
    for layer in bb_layers:
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

        g_model_checkpoint_path = os.path.join(
            logs,
            "chk_pt",
            "Pruning",
            model_arch,
            dataset_name,
            "G",
            f"Prune_type_{prune_type}",
            f"cav_flattening_type_{cav_flattening_type}",
            layer,
            f"epoch_{epochs}_lr_{g_lr}"
        )

        tb_path = os.path.join(
            logs,
            "tensorboard_logs",
            "G",
            "Pruning",
            f"Prune_type_{prune_type}",
            model_arch,
            dataset_name,
            f"cav_flattening_type_{cav_flattening_type}",
            layer
        )

        utils.create_dir(
            path_dict={
                "path_name": g_model_checkpoint_path,
                "path_type": "checkpoint-for-G"
            })
        utils.create_dir(
            path_dict={
                "path_name": tb_path,
                "path_type": "tensorboard-for-G"
            })

        print(g_model_checkpoint_path)
        completeness_scores = []
        for _ite in range(start_iter, prune_iterations):
            print(f"Prune iteration: {_ite} =======================================>")
            concept_vectors = ca_utils.get_concept_vectors_for_pruning(
                _ite,
                cav_path,
                layer,
                cav_flattening_type
            )

            # I took (-lm.coeff) as cav while saving it.
            # So, I need to multiply with -1 with the cavs to get the concept vectors
            torch_concept_vector = torch.from_numpy(concept_vectors).to(device, dtype=torch.float32)
            print(f"CAV size: {torch_concept_vector.size()}")
            bb_model = prun_utils.load_BB_model_w_pruning(
                model_arch,
                num_classes,
                pretrained,
                transfer_learning,
                dataset_name,
                device,
                _ite,
                bb_checkpoint_path
            )
            bb_model_meta = Model_Meta(bb_model, [layer])
            start = time.time()
            train_activation_file = f"train_activations_tensor_prune_iteration_{_ite}.pth.tar"
            train_y_hat_bb_file = f"train_np_y_hat_prune_iteration_{_ite}.npy"
            train_y_gt = f"train_np_y_GT_prune_iteration_{_ite}.npy"

            val_activation_file = f"val_activations_tensor_prune_iteration_{_ite}.pth.tar"
            val_y_hat_bb_file = f"val_np_y_hat_prune_iteration_{_ite}.npy"
            val_y_gt = f"val_np_y_GT_prune_iteration_{_ite}.npy"

            test_activation_file = f"test_activations_tensor_prune_iteration_{_ite}.pth.tar"
            test_y_hat_bb_file = f"test_np_y_hat_prune_iteration_{_ite}.npy"
            test_y_gt = f"test_np_y_GT_prune_iteration_{_ite}.npy"

            train_activations = torch.load(os.path.join(activations_path, train_activation_file))
            train_np_y_hat_bb = np.load(os.path.join(activations_path, train_y_hat_bb_file))
            train_np_y_gt = np.load(os.path.join(activations_path, train_y_gt))

            val_activations = torch.load(os.path.join(activations_path, val_activation_file))
            val_np_y_hat_bb = np.load(os.path.join(activations_path, val_y_hat_bb_file))
            val_np_y_gt = np.load(os.path.join(activations_path, val_y_gt))

            test_activations = torch.load(os.path.join(activations_path, test_activation_file))
            test_np_y_hat_bb = np.load(os.path.join(activations_path, test_y_hat_bb_file))
            test_np_y_gt = np.load(os.path.join(activations_path, test_y_gt))

            print("Sizes: Train, Val, Test")
            print(train_activations.size(), val_activations.size(), test_activations.size())
            print(train_np_y_hat_bb.shape, val_np_y_hat_bb.shape, test_np_y_hat_bb.shape)
            print(train_np_y_gt.shape, val_np_y_gt.shape, test_np_y_gt.shape)

            train_loader = DataLoader(
                TensorDataset(
                    torch.Tensor(train_activations), torch.Tensor(train_np_y_hat_bb), torch.Tensor(train_np_y_gt)
                ),
                batch_size=128,
                num_workers=4,
                shuffle=True
            )

            val_loader = DataLoader(
                TensorDataset(
                    torch.Tensor(val_activations), torch.Tensor(val_np_y_hat_bb), torch.Tensor(val_np_y_gt)
                ),
                batch_size=10,
                shuffle=False
            )

            test_loader = DataLoader(
                TensorDataset(
                    torch.Tensor(test_activations), torch.Tensor(test_np_y_hat_bb), torch.Tensor(test_np_y_gt)
                ),
                batch_size=10,
                shuffle=False
            )

            done = time.time()
            elapsed = done - start
            print("Time to load dataset: " + str(elapsed) + " secs")
            start = time.time()
            completeness_score = cav_train.train_concept_to_activation_model(
                _ite,
                cav_path,
                train_loader,
                val_loader,
                test_loader,
                torch_concept_vector,
                bb_model,
                bb_model_meta,
                model_arch,
                cav_flattening_type,
                dataset_name,
                layer,
                g_lr,
                g_model_checkpoint_path,
                tb_path,
                epochs,
                hidden_features,
                th,
                val_after_th,
                num_labels,
                device
            )

            completeness_scores.append(completeness_score)
            done = time.time()
            elapsed = done - start
            print("Time to train for the iteration: " + str(elapsed) + " secs")

        print("All completeness scores:")
        print(completeness_scores)


def train_G_completeness_w_pruning_derma(
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
        bb_layers,
        num_labels,
        g_lr,
        batch_size,
        epochs,
        hidden_features,
        th,
        val_after_th,
        attribute_file_name,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    layer = "derma"
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
    g_model_checkpoint_path = os.path.join(
            logs,
            "chk_pt",
            "Pruning",
            model_arch,
            dataset_name,
            "G",
            f"Prune_type_{prune_type}",
            f"cav_flattening_type_{cav_flattening_type}",
            layer,
            f"epoch_{epochs}_lr_{g_lr}"
        )

    tb_path = os.path.join(
            logs,
            "tensorboard_logs",
            "G",
            "Pruning",
            f"Prune_type_{prune_type}",
            model_arch,
            dataset_name,
            f"cav_flattening_type_{cav_flattening_type}",
            layer
        )

    utils.create_dir(
            path_dict={
                "path_name": g_model_checkpoint_path,
                "path_type": "checkpoint-for-G"
            })
    utils.create_dir(
            path_dict={
                "path_name": tb_path,
                "path_type": "tensorboard-for-G"
            })

    print(g_model_checkpoint_path)
    completeness_scores = []
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
    train_loader, val_loader, idx_to_class = load_HAM10k_data(
        seed, data_root, transform, class_to_idx, batch_size, mode="train"
    )

    for _ite in range(start_iter, prune_iterations):
        print()
        print(f"*************************** Prune iteration: {_ite} ***************************")
        concepts_dict = pickle.load(
            open(os.path.join(cav_path, f"derma_ham10000_0.01_50_ite_{_ite}.pkl"), "rb")
        )
        model, model_bottom, model_top = prun_utils.load_BB_model_w_pruning_derma(
            bb_dir,
            device,
            _ite,
            bb_checkpoint_path
        )
        start = time.time()
        completeness_score = cav_train.train_concept_to_activation_model_derma(
            _ite,
            concepts_dict,
            train_loader,
            val_loader,
            model,
            model_bottom,
            model_top,
            model_arch,
            cav_flattening_type,
            dataset_name,
            layer,
            g_lr,
            g_model_checkpoint_path,
            tb_path,
            epochs,
            hidden_features,
            th,
            val_after_th,
            num_labels,
            device
        )

        completeness_scores.append(completeness_score)
        done = time.time()
        elapsed = done - start
        print("Time to train for the iteration: " + str(elapsed) + " secs")

    print("All completeness scores:")
    print(completeness_scores)
