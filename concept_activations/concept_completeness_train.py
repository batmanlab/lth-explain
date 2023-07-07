import copy
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import concept_activations.concept_activations_utils as ca_utils
import utils
from concept_activations.flatten_LR import Flatten_LR
from concept_activations.g import G
from run_manager import RunManager


def train_concept_to_activation_model(
        prune_ite,
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
        bb_layer,
        g_lr,
        g_model_checkpoint_path,
        tb_path,
        epochs,
        hidden_features,
        th,
        val_after_th,
        num_labels,
        device
):
    best_completeness_score = 0
    final_parameters = OrderedDict(
        epoch=[epochs],
        layer=[bb_layer],
        dataset=[dataset_name],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')],
        cav_flattening_type=[cav_flattening_type],
        prune_iter=[prune_ite]
    )
    run_id = utils.get_runs(final_parameters)[0]

    g_model_ip_size, g_model_op_size = ca_utils.get_g_model_ip_op_size(
        train_loader,
        device,
        bb_model,
        bb_model_meta,
        torch_concept_vector,
        bb_layer,
        cav_flattening_type,
        dataset_name
    )

    print(f"g_model input size: {g_model_ip_size}")
    print(f"g_model output size: {g_model_op_size}")

    for param in bb_model.parameters():
        param.requires_grad = False

    g = G(g_model_ip_size, g_model_op_size, hidden_features).to(device)
    bb_model_mid, bb_model_tail = ca_utils.dissect_bb_model(model_arch, bb_model)
    optimizer = torch.optim.Adam(g.parameters(), lr=g_lr)
    criterion = torch.nn.CrossEntropyLoss()
    bb_model.eval()

    t = Flatten_LR(ip_size=2048, op_size=108).to(device)
    checkpoint_t = os.path.join(cav_path, f"adaptive_avg_pooled_train_model_prune_iteration_{prune_ite}.pth.tar")
    t.load_state_dict(torch.load(checkpoint_t))
    t.eval()
    print(torch_concept_vector[0])
    torch_concept_vector = t.model[0].weight.detach()
    print(f"Cav details: ")
    print(torch_concept_vector.size())
    print(torch_concept_vector[0])
    run_manager = RunManager(prune_ite, g_model_checkpoint_path, tb_path, train_loader, val_loader)
    run_manager.begin_run(run_id)
    for epoch in range(epochs):
        run_manager.begin_epoch()
        g.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (activations, _, labels) in enumerate(train_loader):
                bs = activations.size(0)
                activations = activations.to(device)
                labels = labels.to(torch.long).to(device)
                norm_vc = ca_utils.get_normalized_vc(
                    activations,
                    torch_concept_vector,
                    th,
                    val_after_th,
                    cav_flattening_type
                )
                concept_to_act = g(norm_vc)
                concept_to_pred = ca_utils.get_concept_to_pred(
                    concept_to_act,
                    bs,
                    activations,
                    bb_model_mid,
                    bb_model_tail
                )
                optimizer.zero_grad()
                train_loss = criterion(concept_to_pred, labels)
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(
                    concept_to_pred, labels, num_classes=concept_to_pred.size(1)
                )

                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        g.eval()
        out_put_predict_bb = torch.FloatTensor().cuda()
        out_put_predict_bb_1 = torch.FloatTensor().cuda()
        out_put_GT = torch.FloatTensor().cuda()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (activations, input_to_pred, labels) in enumerate(val_loader):
                    bs = activations.size(0)
                    activations = activations.to(device)
                    input_to_pred = input_to_pred.to(device)
                    labels = labels.to(torch.long).to(device)
                    out_put_predict_bb = torch.cat((out_put_predict_bb, input_to_pred), dim=0)
                    out_put_GT = torch.cat((out_put_GT, labels), dim=0)

                    input_to_pred_1 = bb_model_tail(
                        torch.nn.AdaptiveAvgPool2d((1, 1))(activations).reshape(-1, activations.size(1) * 1 * 1)
                    )
                    out_put_predict_bb_1 = torch.cat((out_put_predict_bb_1, input_to_pred_1), dim=0)

                    norm_vc = ca_utils.get_normalized_vc(
                        activations,
                        torch_concept_vector,
                        th,
                        val_after_th,
                        cav_flattening_type
                    )
                    concept_to_act = g(norm_vc)
                    concept_to_pred = ca_utils.get_concept_to_pred(
                        concept_to_act,
                        bs,
                        activations,
                        bb_model_mid,
                        bb_model_tail
                    )
                    val_loss = criterion(concept_to_pred, labels)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(
                        concept_to_pred, labels, num_classes=concept_to_pred.size(1)
                    )
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(g)
        bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
        bb_acc_1 = utils.get_correct(out_put_predict_bb_1, out_put_GT, num_labels) / out_put_GT.size(0)
        print(bb_acc, bb_acc_1, run_manager.get_final_val_accuracy())
        epoch_completeness_score = utils.cal_completeness_score(
            num_labels,
            run_manager.get_final_val_accuracy(),
            bb_acc
        )
        best_completeness_score = utils.cal_completeness_score(
            num_labels,
            run_manager.get_final_best_val_accuracy(),
            bb_acc
        )
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} "
              f"Val_Accuracy: {round(run_manager.get_final_val_accuracy(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} "
              f"Epoch_completeness: {round(epoch_completeness_score, 4)} "
              f"Best_completeness: {round(best_completeness_score, 4)} "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()

    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_bb_1 = torch.FloatTensor().cuda()
    out_put_predict_g = torch.FloatTensor().cuda()
    out_put_GT = torch.FloatTensor().cuda()
    g.eval()
    g.load_state_dict(torch.load(os.path.join(g_model_checkpoint_path, f"best_prune_iteration_{prune_ite}.pth.tar")))
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, (activations, input_to_pred, labels) in enumerate(test_loader):
                bs = activations.size(0)
                activations = activations.to(device)
                input_to_pred = input_to_pred.to(device)
                labels = labels.to(torch.long).to(device)
                out_put_predict_bb = torch.cat((out_put_predict_bb, input_to_pred), dim=0)
                out_put_GT = torch.cat((out_put_GT, labels), dim=0)
                input_to_pred_1 = bb_model_tail(
                    torch.nn.AdaptiveAvgPool2d((1, 1))(activations).reshape(-1, activations.size(1) * 1 * 1)
                )
                out_put_predict_bb_1 = torch.cat((out_put_predict_bb_1, input_to_pred_1), dim=0)
                norm_vc = ca_utils.get_normalized_vc(
                    activations,
                    torch_concept_vector,
                    th,
                    val_after_th,
                    cav_flattening_type
                )
                concept_to_act = g(norm_vc)
                concept_to_pred = ca_utils.get_concept_to_pred(
                    concept_to_act,
                    bs,
                    activations,
                    bb_model_mid,
                    bb_model_tail
                )
                out_put_predict_g = torch.cat((out_put_predict_g, concept_to_pred), dim=0)
                t.set_postfix(batch_idepoch='{0}'.format(batch_id))
                t.update()
    bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
    bb_acc_1 = utils.get_correct(out_put_predict_bb_1, out_put_GT, num_labels) / out_put_GT.size(0)
    g_acc = utils.get_correct(out_put_predict_g, out_put_GT, num_labels) / out_put_GT.size(0)

    print("Test stats: ")
    print(f"bb_acc: {bb_acc}, bb_acc_1: {bb_acc_1}, g_acc: {g_acc}")
    test_completeness_score = utils.cal_completeness_score(
        num_labels,
        g_acc,
        bb_acc
    )
    print(f"test_completeness_score: {test_completeness_score}")
    return test_completeness_score


def train_concept_to_activation_model_derma(
        prune_ite,
        concepts_dict,
        train_loader,
        val_loader,
        model,
        model_bottom,
        model_top,
        model_arch,
        cav_flattening_type,
        dataset_name,
        bb_layer,
        g_lr,
        g_model_checkpoint_path,
        tb_path,
        epochs,
        hidden_features,
        th,
        val_after_th,
        num_labels,
        device
):
    residual = copy.deepcopy(model_top)
    residual.eval()
    cavs = []
    for key in concepts_dict.keys():
        cavs.append(concepts_dict[key][0][0].tolist())
    cavs = np.array(cavs)
    torch_concept_vector = torch.from_numpy(cavs).to(device, dtype=torch.float32)
    best_completeness_score = 0
    final_parameters = OrderedDict(
        epoch=[epochs],
        layer=[bb_layer],
        dataset=[dataset_name],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')],
        cav_flattening_type=[cav_flattening_type],
        prune_iter=[prune_ite]
    )
    run_id = utils.get_runs(final_parameters)[0]

    for param in model.parameters():
        param.requires_grad = False

    g = G(8, 2048, hidden_features).to(device)
    optimizer = torch.optim.Adam(g.parameters(), lr=g_lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    run_manager = RunManager(prune_ite, g_model_checkpoint_path, tb_path, train_loader, val_loader)
    run_manager.begin_run(run_id)
    for epoch in range(epochs):
        run_manager.begin_epoch()
        g.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, (train_images, train_y) in enumerate(train_loader):
                train_images = train_images.to(device)
                train_y = train_y.to(torch.long).to(device)
                bs = train_images.size(0)
                with torch.no_grad():
                    bb_logits = model(train_images)
                    feature_x = model_bottom(train_images)
                norm_vc = ca_utils.get_normalized_vc(
                    feature_x,
                    torch_concept_vector,
                    th,
                    val_after_th,
                    cav_flattening_type="flattened"
                )
                concept_to_act = g(norm_vc)
                completeness_logits = residual(concept_to_act)
                optimizer.zero_grad()
                train_loss = criterion(completeness_logits, train_y)
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(
                    completeness_logits, train_y, num_classes=completeness_logits.size(1)
                )

                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        g.eval()
        out_put_predict_bb = torch.FloatTensor().cuda()
        out_put_predict_bb_1 = torch.FloatTensor().cuda()
        out_put_GT = torch.FloatTensor().cuda()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, (val_images, val_y) in enumerate(val_loader):
                    val_images = val_images.to(device)
                    val_y = val_y.to(torch.long).to(device)
                    bs = val_images.size(0)
                    with torch.no_grad():
                        bb_logits = model(val_images)
                        feature_x = model_bottom(val_images)
                    input_to_pred = bb_logits
                    out_put_predict_bb = torch.cat((out_put_predict_bb, input_to_pred), dim=0)
                    out_put_GT = torch.cat((out_put_GT, val_y), dim=0)

                    norm_vc = ca_utils.get_normalized_vc(
                        feature_x,
                        torch_concept_vector,
                        th,
                        val_after_th,
                        cav_flattening_type="flattened"
                    )
                    concept_to_act = g(norm_vc)
                    completeness_logits = residual(concept_to_act)
                    val_loss = criterion(completeness_logits, val_y)

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(
                        completeness_logits, val_y, num_classes=completeness_logits.size(1)
                    )
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss))
                    t.update()

        run_manager.end_epoch(g)
        bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
        print(bb_acc, run_manager.get_final_val_accuracy())
        epoch_completeness_score = utils.cal_completeness_score(
            num_labels,
            run_manager.get_final_val_accuracy(),
            bb_acc
        )
        best_completeness_score = utils.cal_completeness_score(
            num_labels,
            run_manager.get_final_best_val_accuracy(),
            bb_acc
        )
        print(f"Epoch: [{epoch + 1}/{epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} "
              f"Val_Accuracy: {round(run_manager.get_final_val_accuracy(), 4)} "
              f"Best_Val_Accuracy: {round(run_manager.get_final_best_val_accuracy(), 4)} "
              f"Epoch_completeness: {round(epoch_completeness_score, 4)} "
              f"Best_completeness: {round(best_completeness_score, 4)} "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()

    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_g = torch.FloatTensor().cuda()
    out_put_GT = torch.FloatTensor().cuda()
    g.eval()
    g.load_state_dict(torch.load(os.path.join(g_model_checkpoint_path, f"best_prune_iteration_{prune_ite}.pth.tar")))
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as t:
            for batch_id, (val_images, val_y) in enumerate(val_loader):
                val_images = val_images.to(device)
                val_y = val_y.to(torch.long).to(device)
                bs = val_images.size(0)
                with torch.no_grad():
                    bb_logits = model(val_images)
                    feature_x = model_bottom(val_images)
                input_to_pred = bb_logits
                out_put_predict_bb = torch.cat((out_put_predict_bb, input_to_pred), dim=0)
                out_put_GT = torch.cat((out_put_GT, val_y), dim=0)

                norm_vc = ca_utils.get_normalized_vc(
                    feature_x,
                    torch_concept_vector,
                    th,
                    val_after_th,
                    cav_flattening_type="flattened"
                )
                concept_to_act = g(norm_vc)
                completeness_logits = residual(concept_to_act)

                out_put_predict_g = torch.cat((out_put_predict_g, completeness_logits), dim=0)
                t.set_postfix(batch_idepoch='{0}'.format(batch_id))
                t.update()
    bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
    g_acc = utils.get_correct(out_put_predict_g, out_put_GT, num_labels) / out_put_GT.size(0)

    print("Test stats: ")
    print(f"bb_acc: {bb_acc}, g_acc: {g_acc}")
    test_completeness_score = utils.cal_completeness_score(
        num_labels,
        g_acc,
        bb_acc
    )

    print(f"***************************** prune_ite: {prune_ite}, bb_acc: {bb_acc}, g_acc: {g_acc}, "
          f"test_completeness_score: {test_completeness_score} *****************************")
    print()
    return test_completeness_score
