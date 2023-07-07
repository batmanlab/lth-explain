import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import concept_activations.concept_activations_utils as ca_utils
import concept_activations.concept_completeness_mnist_test as cav_test
import lth_pruning.pruning_utils as pruning_utils
import utils
from model_factory.model_meta import Model_Meta


def test_pruned_model_with_completeness_score(
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
        batch_size,
        epochs,
        g_lr,
        hidden_features,
        th,
        val_after_th,
        attribute_file_name,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform_params = {
        "img_size": img_size
    }
    # start = time.time()
    # test_loader = pruning_utils.get_test_dataloader(
    #     dataset_name,
    #     data_root,
    #     json_root,
    #     batch_size,
    #     transform_params
    # )
    # done = time.time()
    # elapsed = done - start
    # print("Time to load the test dataset from disk: " + str(elapsed) + " secs")
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
            layer,
            f"epoch_{epochs}_lr_{g_lr}"
        )
        utils.create_dir(
            path_dict={
                "path_name": prune_stat_path,
                "path_type": "prune_stat_path-for-each-prune-iteration"
            })

        percent_weight_remaining = [
            100.0, 90.0, 81.0, 72.9, 65.6, 59.1, 53.2, 47.8, 43.1, 38.8, 34.9, 31.4,
            28.3, 25.4, 22.9, 20.6, 18.6, 16.7, 15.0, 13.5, 12.2, 11.0, 9.9, 8.9,
            8.0, 7.2, 6.5, 5.9, 5.3, 4.7, 4.3, 3.9, 3.5, 3.1, 2.8
        ]

        metric_arr = []
        g_acc_arr = []
        completeness_score_arr =[]
        for _ite in range(start_iter, prune_iterations):
            print(f"Prune iteration: {_ite} =======================================>")
            concept_vectors = ca_utils.get_concept_vectors_for_pruning(
                _ite,
                cav_path,
                layer,
                cav_flattening_type
            )

            torch_concept_vector = torch.from_numpy(concept_vectors).to(device, dtype=torch.float32)
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
            g_model_checkpoint = os.path.join(g_model_checkpoint_path, f"best_prune_iteration_{_ite}.pth.tar")
            print("Checkpoint for G loaded from here:\n")
            print(g_model_checkpoint)
            stats = cav_test.calculate_concept_completeness_score(
                _ite,
                cav_path,
                test_loader,
                torch_concept_vector,
                bb_model,
                bb_model_meta,
                model_arch,
                cav_flattening_type,
                layer,
                g_model_checkpoint,
                hidden_features,
                th,
                val_after_th,
                num_labels,
                percent_weight_remaining[_ite],
                dataset_name,
                device
            )

            metric = stats["metric"]
            metric_arr.append(metric)
            g_acc_arr.append(stats["g_acc"])
            completeness_score_arr.append(stats["completeness_score"])
            print(f"Percent weight remaining: {metric['percent_weight_remaining']}")
            print("Accuracy using BB: ")
            print(f"Accuracy: {metric['BB']['Accuracy']}")
            print(f"F1 score: {metric['BB']['F1_score']}")

            print("Accuracy using G: ")
            print(f"Accuracy: {metric['G']['Accuracy']}")
            print(f"F1 score: {metric['G']['F1_score']}")

            print(f"Completeness score for dataset [{dataset_name}] using [{model_arch}]: "
                  f"{metric['Completeness_score']}")

            np.save(
                os.path.join(prune_stat_path, f"out_put_GT_prune_ite_{_ite}.npy"),
                stats["out_put_GT_np"]
            )
            np.save(
                os.path.join(prune_stat_path, f"out_put_predict_bb_prune_ite_{_ite}.npy"),
                stats["out_put_predict_bb_np"]
            )
            np.save(
                os.path.join(prune_stat_path, f"out_put_predict_g_prune_ite_{_ite}.npy"),
                stats["out_put_predict_g"]
            )

            done = time.time()
            elapsed = done - start
            print("Time to execute for this iteration: " + str(elapsed) + " secs")

        metric_file = open(os.path.join(
            prune_stat_path,
            f"metric_{cav_flattening_type}_completeness.pkl"
        ), "wb")
        pickle.dump(metric_arr, metric_file)
        metric_file.close()
        print(f"Activation dictionary is saved in the location: {prune_stat_path}")
        print(completeness_score_arr)
        print(g_acc_arr)
