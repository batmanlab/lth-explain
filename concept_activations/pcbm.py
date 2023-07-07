import os
from collections import OrderedDict
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm

import utils
from concept_activations.flatten_LR import Flatten_LR
from concept_activations.pcbm_model import PCBM
from run_manager import RunManager


def train_eval_pcbm(
        prune_ite,
        cav_path,
        train_loader,
        val_loader,
        test_loader,
        torch_concept_vector,
        cav_flattening_type,
        dataset_name,
        bb_layer,
        pcbm_lr,
        pcbm_alpha,
        pcbm_l1_ratio,
        pcbm_model_checkpoint_path,
        tb_path,
        epochs,
        num_labels,
        device
):
    final_parameters = OrderedDict(
        epoch=[epochs],
        layer=[bb_layer],
        dataset=[dataset_name],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')],
        cav_flattening_type=[cav_flattening_type],
        prune_iter=[prune_ite]
    )
    run_id = utils.get_runs(final_parameters)[0]

    pcbm = PCBM(ip_size=108, op_size=200).to(device)
    optimizer = torch.optim.Adam(pcbm.parameters(), lr=pcbm_lr)
    criterion = torch.nn.CrossEntropyLoss()

    projection = Flatten_LR(ip_size=2048, op_size=108).to(device)
    checkpoint_t = os.path.join(cav_path, f"adaptive_avg_pooled_train_model_prune_iteration_{prune_ite}.pth.tar")
    projection.load_state_dict(torch.load(checkpoint_t))
    projection.eval()

    run_manager = RunManager(prune_ite, pcbm_model_checkpoint_path, tb_path, train_loader, val_loader)
    run_manager.begin_run(run_id)
    for epoch in range(epochs):
        run_manager.begin_epoch()
        pcbm.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (activations, _, labels) in enumerate(train_loader):
                bs = activations.size(0)
                activations = activations.to(device)
                labels = labels.to(torch.long).to(device)
                avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                flatten_activations = avgpool(activations).view(bs, -1)
                pred_pcbm = pcbm(projection(flatten_activations))
                optimizer.zero_grad()
                train_loss = criterion(pred_pcbm, labels) + pcbm_alpha * (
                        pcbm_l1_ratio * pcbm.model.weight.abs().sum() + (1 - pcbm_l1_ratio) * pcbm.model.weight.pow(
                    2).sum())
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(
                    pred_pcbm, labels, num_classes=pred_pcbm.size(1)
                )

                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        pcbm.eval()
        out_put_predict_bb = torch.FloatTensor().cuda()
        out_put_GT = torch.FloatTensor().cuda()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (activations, pred_bb, labels) in enumerate(val_loader):
                    bs = activations.size(0)
                    activations = activations.to(device)
                    pred_bb = pred_bb.to(device)
                    labels = labels.to(torch.long).to(device)
                    out_put_predict_bb = torch.cat((out_put_predict_bb, pred_bb), dim=0)
                    out_put_GT = torch.cat((out_put_GT, labels), dim=0)
                    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                    flatten_activations = avgpool(activations).view(bs, -1)
                    pred_pcbm = pcbm(projection(flatten_activations))
                    val_loss = criterion(pred_pcbm, labels) + pcbm_alpha * (
                            pcbm_l1_ratio * pcbm.model.weight.abs().sum() + (1 - pcbm_l1_ratio) * pcbm.model.weight.pow(
                        2).sum())

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(
                        pred_pcbm, labels, num_classes=pred_pcbm.size(1)
                    )
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(pcbm)
        bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"bb_acc: {bb_acc} "
              f"Train_PCBM_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} "
              f"Val__PCBM_Accuracy: {round(run_manager.get_final_val_accuracy(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()

    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_pcbm = torch.FloatTensor().cuda()
    out_put_GT = torch.FloatTensor().cuda()
    pcbm.eval()
    pcbm.load_state_dict(
        torch.load(os.path.join(pcbm_model_checkpoint_path, f"best_prune_iteration_{prune_ite}.pth.tar"))
    )
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, (activations, input_to_pred, labels) in enumerate(test_loader):
                bs = activations.size(0)
                activations = activations.to(device)
                input_to_pred = input_to_pred.to(device)
                labels = labels.to(torch.long).to(device)
                avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                flatten_activations = avgpool(activations).view(bs, -1)
                pred_pcbm = pcbm(projection(flatten_activations))

                out_put_predict_bb = torch.cat((out_put_predict_bb, input_to_pred), dim=0)
                out_put_GT = torch.cat((out_put_GT, labels), dim=0)
                out_put_predict_pcbm = torch.cat((out_put_predict_pcbm, pred_pcbm), dim=0)

                t.set_postfix(batch_idepoch='{0}'.format(batch_id))
                t.update()
    bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
    pcbm_acc = utils.get_correct(out_put_predict_pcbm, out_put_GT, num_labels) / out_put_GT.size(0)

    print("Test stats: ")
    print(f"bb_acc: {bb_acc}, pcbm_acc: {pcbm_acc}")
    return pcbm_acc


def train_eval_pcbm_derma(
        prune_ite,
        cav_path,
        concept_bank,
        model,
        model_bottom,
        model_top,
        train_loader,
        val_loader,
        cav_flattening_type,
        dataset_name,
        bb_layer,
        pcbm_lr,
        pcbm_alpha,
        pcbm_l1_ratio,
        pcbm_model_checkpoint_path,
        tb_path,
        epochs,
        num_labels,
        device
):
    final_parameters = OrderedDict(
        epoch=[epochs],
        layer=[bb_layer],
        dataset=[dataset_name],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')],
        cav_flattening_type=[cav_flattening_type],
        prune_iter=[prune_ite]
    )
    run_id = utils.get_runs(final_parameters)[0]

    pcbm = PCBM(ip_size=8, op_size=2).to(device)
    optimizer = torch.optim.Adam(pcbm.parameters(), lr=pcbm_lr)
    criterion = torch.nn.CrossEntropyLoss()
    run_manager = RunManager(prune_ite, pcbm_model_checkpoint_path, tb_path, train_loader, val_loader)
    run_manager.begin_run(run_id)

    for epoch in range(epochs):
        run_manager.begin_epoch()
        pcbm.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (images, target) in enumerate(train_loader):
                images = images.to(device)
                target = target.to(torch.long).to(device)
                bs = images.size(0)
                with torch.no_grad():
                    bb_logits = model(images)
                    phi = model_bottom(images)
                    train_concepts = compute_dist(concept_bank, phi)
                pred_pcbm = pcbm(train_concepts)
                optimizer.zero_grad()
                train_loss = criterion(pred_pcbm, target) + pcbm_alpha * (
                        pcbm_l1_ratio * pcbm.model.weight.abs().sum() + (1 - pcbm_l1_ratio) * pcbm.model.weight.pow(
                    2).sum())
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(
                    pred_pcbm, target, num_classes=pred_pcbm.size(1)
                )

                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        pcbm.eval()
        out_put_predict_bb = torch.FloatTensor().cuda()
        out_put_GT = torch.FloatTensor().cuda()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (images, target) in enumerate(val_loader):
                    images = images.to(device)
                    target = target.to(torch.long).to(device)
                    bs = images.size(0)
                    with torch.no_grad():
                        bb_logits = model(images)
                        phi = model_bottom(images)
                        val_concepts = compute_dist(concept_bank, phi)
                        out_put_predict_bb = torch.cat((out_put_predict_bb, bb_logits), dim=0)
                        out_put_GT = torch.cat((out_put_GT, target), dim=0)
                    pred_pcbm = pcbm(val_concepts)
                    val_loss = criterion(pred_pcbm, target) + pcbm_alpha * (
                            pcbm_l1_ratio * pcbm.model.weight.abs().sum() + (1 - pcbm_l1_ratio) * pcbm.model.weight.pow(
                        2).sum())

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(
                        pred_pcbm, target, num_classes=pred_pcbm.size(1)
                    )
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(pcbm)
        bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"bb_acc: {bb_acc} "
              f"Train_PCBM_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} "
              f"Val__PCBM_Accuracy: {round(run_manager.get_final_val_accuracy(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()

    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_pcbm = torch.FloatTensor().cuda()
    out_put_GT = torch.FloatTensor().cuda()
    pcbm.eval()
    pcbm.load_state_dict(
        torch.load(os.path.join(pcbm_model_checkpoint_path, f"best_prune_iteration_{prune_ite}.pth.tar"))
    )
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as t:
            for batch_id, (images, target) in enumerate(val_loader):
                images = images.to(device)
                target = target.to(torch.long).to(device)
                bs = images.size(0)
                with torch.no_grad():
                    bb_logits = model(images)
                    phi = model_bottom(images)
                    val_concepts = compute_dist(concept_bank, phi)
                pred_pcbm = pcbm(val_concepts)

                out_put_predict_bb = torch.cat((out_put_predict_bb, bb_logits), dim=0)
                out_put_GT = torch.cat((out_put_GT, target), dim=0)
                out_put_predict_pcbm = torch.cat((out_put_predict_pcbm, pred_pcbm), dim=0)

                t.set_postfix(batch_idepoch='{0}'.format(batch_id))
                t.update()
    bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
    pcbm_acc = utils.get_correct(out_put_predict_pcbm, out_put_GT, num_labels) / out_put_GT.size(0)

    print("Test stats: ")
    print(f"bb_acc: {bb_acc}, pcbm_acc: {pcbm_acc}")
    return pcbm_acc


def compute_dist(concept_bank, phi):
    margins = (torch.matmul(concept_bank.vectors, phi.T) + concept_bank.intercepts) / concept_bank.norms
    return margins.T


def get_pcbm_concepts(
        prune_ite,
        pcbm_model_checkpoint_path,
        num_labels,
        class_list,
        concept_names,
        topK,
        device
):
    pcbm = PCBM(ip_size=len(concept_names), op_size=num_labels).to(device)
    pcbm.eval()
    pcbm.load_state_dict(
        torch.load(os.path.join(pcbm_model_checkpoint_path, f"best_prune_iteration_{prune_ite}.pth.tar"))
    )
    params = pcbm.model.weight
    class_idx_list = []
    class_name_list = []
    topK_concept_index_list = []
    topK_concept_name_list = []
    with torch.no_grad():
        for idx in range(num_labels):
            concept_weight = params[idx, :]
            top_concepts = torch.topk(concept_weight, topK)[1].tolist()
            concepts_K = list(map(lambda i: concept_names[i], top_concepts))
            class_idx_list.append(idx)
            class_name_list.append(class_list[idx])
            topK_concept_index_list.append(top_concepts)
            topK_concept_name_list.append(concepts_K)

    concepts_results = {
        "class_idx": class_idx_list,
        "class_name": class_name_list,
        "top_concept_index": topK_concept_index_list,
        "top_concept_names": topK_concept_name_list
    }

    df = pd.DataFrame(concepts_results)
    return df