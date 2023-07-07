import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset.dataset_attributes_mnist import Dataset_attributes_mnist
from dataset.dataset_cubs import Dataset_cub
from dataset.dataset_utils import get_transforms, get_dataset_with_attributes, \
    get_dataset_with_image_and_attributes, get_transform_cub
from lth_pruning.pruning_utils import load_BB_model_w_pruning, get_image_labels
from model_factory.model_meta import Model_Meta


def create_activation_DB(dataloader, bb_layer, device, model, model_meta, dataset_name):
    features_tensor = torch.FloatTensor()
    attr_GT = torch.FloatTensor().cuda()
    y_GT_tensor = torch.FloatTensor().cuda()
    y_hat_tensor = torch.FloatTensor().cuda()
    activations = {}
    activations[bb_layer] = []

    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            for batch_id, data_tuple in enumerate(dataloader):
                image, label, attribute = data_tuple
                image = image.to(device)
                label = label.to(device)
                attribute = attribute.to(device)
                y_hat = model(image)
                features = model_meta.model_activations_store[bb_layer].cpu()
                features_tensor = torch.cat((features_tensor, features), dim=0)
                attr_GT = torch.cat((attr_GT, attribute), dim=0)
                y_GT_tensor = torch.cat((y_GT_tensor, label), dim=0)
                y_hat_tensor = torch.cat((y_hat_tensor, y_hat), dim=0)
                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    print("Activations are generated..")
    activations[bb_layer] = features_tensor.numpy()
    print(activations[bb_layer].shape)

    print(features_tensor.cpu().size())
    print(attr_GT.cpu().numpy().shape)
    print(y_GT_tensor.cpu().numpy().shape)
    print(y_hat_tensor.cpu().numpy().shape)
    bb_acc = utils.get_correct(y_hat_tensor, y_GT_tensor, y_hat_tensor.size(1)) / y_GT_tensor.size(0)
    print(f"BB Acc: {bb_acc}")
    return (
        activations, attr_GT.cpu().numpy(),  y_GT_tensor.cpu().numpy(),  y_hat_tensor.cpu().numpy(),
        features_tensor.cpu()
    )


def get_dataloaders(dataset_name, img_size, data_root, json_root, batch_size, attribute_file_name):
    if dataset_name == "mnist":
        return load_datasets_mnist(
            img_size,
            data_root,
            json_root,
            dataset_name,
            batch_size,
            attribute_file_name
        )
    elif dataset_name == "cub":
        return load_datasets_cub(
            img_size,
            data_root,
            json_root,
            dataset_name,
            batch_size,
            attribute_file_name
        )


def load_datasets_cub(img_size, data_root, json_root, dataset_name, batch_size, attribute_file_name):
    start = time.time()

    train_transform = get_transform_cub(size=img_size, data_augmentation=True)
    train_set, train_attributes = get_dataset_with_image_and_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="train",
        attribute_file=attribute_file_name
    )

    train_dataset = Dataset_cub(train_set, train_attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the train dataset from disk: " + str(elapsed) + " secs")

    start = time.time()
    val_transform = get_transform_cub(size=img_size, data_augmentation=False)
    val_set, val_attributes = get_dataset_with_image_and_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="val",
        attribute_file=attribute_file_name
    )

    val_dataset = Dataset_cub(val_set, val_attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the val dataset from disk: " + str(elapsed) + " secs")

    start = time.time()
    test_transform = get_transform_cub(size=img_size, data_augmentation=False)
    test_set, test_attributes = get_dataset_with_image_and_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="test",
        attribute_file=attribute_file_name
    )

    test_dataset = Dataset_cub(test_set, test_attributes, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")

    return train_dataloader, val_dataloader, test_dataloader


def load_datasets_mnist(img_size, data_root, json_root, dataset_name, batch_size, attribute_file_name):
    start = time.time()
    transform = get_transforms(size=img_size)
    train_set, train_attributes = get_dataset_with_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="train",
        attribute_file=attribute_file_name
    )

    train_dataset = Dataset_attributes_mnist(train_set, train_attributes, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the train dataset from disk: " + str(elapsed) + " secs")

    start = time.time()
    val_set, val_attributes = get_dataset_with_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="val",
        attribute_file=attribute_file_name
    )

    val_dataset = Dataset_attributes_mnist(val_set, val_attributes, transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the val dataset from disk: " + str(elapsed) + " secs")

    start = time.time()
    test_set, test_attributes = get_dataset_with_attributes(
        data_root=data_root,
        json_root=json_root,
        dataset_name=dataset_name,
        mode="test",
        attribute_file=attribute_file_name
    )

    test_dataset = Dataset_attributes_mnist(test_set, test_attributes, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    done = time.time()
    elapsed = done - start
    print("Time to load the test dataset from disk: " + str(elapsed) + " secs")

    return train_dataloader, val_dataloader, test_dataloader


def save_activations_with_Pruning(
        seed,
        num_classes,
        pretrained,
        transfer_learning,
        prune_type,
        dataset_name,
        data_root,
        json_root,
        batch_size,
        img_size,
        start_iter,
        prune_iterations,
        logs,
        model_arch,
        bb_layers,
        attribute_file_name,
        device
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_path = os.path.join(logs, "chk_pt", "Pruning", model_arch, dataset_name)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            dataset_name,
            img_size,
            data_root,
            json_root,
            batch_size,
            attribute_file_name,
        )
    for layer in bb_layers:
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
        utils.create_dir(
            path_dict={
                "path_name": activations_path,
                "path_type": "activations-of-BB"
            })

        for _ite in range(start_iter, prune_iterations):
            print(f"Prune iteration: {_ite} =======================================>")
            bb_model = load_BB_model_w_pruning(
                model_arch,
                num_classes,
                pretrained,
                transfer_learning,
                dataset_name,
                device,
                _ite,
                checkpoint_path
            )

            bb_model.eval()
            model_meta = Model_Meta(bb_model, bb_layers)
            start = time.time()
            (
                train_activations, train_np_attr_GT, train_np_y,  train_np_y_hat, train_features_tensor
            ) = create_activation_DB(
                train_dataloader,
                layer,
                device,
                bb_model,
                model_meta,
                dataset_name
            )

            utils.save_activations(
                activations_path,
                f"train_activations_prune_iteration_{_ite}.h5",
                layer,
                train_activations
            )
            np.save(os.path.join(activations_path, f"train_np_attr_GT_prune_iteration_{_ite}.npy"), train_np_attr_GT)
            np.save(os.path.join(activations_path, f"train_np_y_GT_prune_iteration_{_ite}.npy"), train_np_y)
            np.save(os.path.join(activations_path, f"train_np_y_hat_prune_iteration_{_ite}.npy"), train_np_y_hat)
            torch.save(
                train_features_tensor,
                os.path.join(activations_path, f"train_activations_tensor_prune_iteration_{_ite}.pth.tar")
            )
            done = time.time()
            elapsed = done - start
            print("Time to create train activations: " + str(elapsed) + " secs")

            start = time.time()
            (
                val_activations, val_np_attr_GT, val_np_y,  val_np_y_hat, val_features_tensor
             ) = create_activation_DB(
                val_dataloader,
                layer,
                device,
                bb_model,
                model_meta,
                dataset_name
            )
            utils.save_activations(
                activations_path,
                f"val_activations_prune_iteration_{_ite}.h5",
                layer,
                val_activations
            )
            np.save(os.path.join(activations_path, f"val_np_attr_GT_prune_iteration_{_ite}.npy"), val_np_attr_GT)
            np.save(os.path.join(activations_path, f"val_np_y_GT_prune_iteration_{_ite}.npy"), val_np_y)
            np.save(os.path.join(activations_path, f"val_np_y_hat_prune_iteration_{_ite}.npy"), val_np_y_hat)
            torch.save(
                val_features_tensor,
                os.path.join(activations_path, f"val_activations_tensor_prune_iteration_{_ite}.pth.tar")
            )
            done = time.time()
            elapsed = done - start
            print("Time to create val activations: " + str(elapsed) + " secs")

            start = time.time()
            (
                test_activations, test_np_attr_GT, test_np_y,  test_np_y_hat, test_features_tensor
             ) = create_activation_DB(
                test_dataloader,
                layer,
                device,
                bb_model,
                model_meta,
                dataset_name
            )
            utils.save_activations(
                activations_path,
                f"test_activations_prune_iteration_{_ite}.h5",
                layer,
                test_activations
            )
            np.save(os.path.join(activations_path, f"test_np_attr_GT_prune_iteration_{_ite}.npy"), test_np_attr_GT)
            np.save(os.path.join(activations_path, f"test_np_y_GT_prune_iteration_{_ite}.npy"), test_np_y)
            np.save(os.path.join(activations_path, f"test_np_y_hat_prune_iteration_{_ite}.npy"), test_np_y_hat)
            torch.save(
                test_features_tensor,
                os.path.join(activations_path, f"test_activations_tensor_prune_iteration_{_ite}.pth.tar")
            )
            done = time.time()
            elapsed = done - start
            print("Time to create test activations: " + str(elapsed) + " secs")
