import os
import sys

from tqdm import tqdm

import concept_activations.concept_activations_utils as ca_utils
from concept_activations.flatten_LR import Flatten_LR
from concept_activations.g import G

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/Project_Pruning"))
import utils
import torch


def calculate_concept_completeness_score(
        prune_ite,
        cav_path,
        test_loader,
        torch_concept_vector,
        bb_model,
        bb_model_meta,
        model_arch,
        cav_flattening_type,
        bb_layer,
        g_model_checkpoint,
        hidden_features,
        th,
        val_after_th,
        num_labels,
        percent_weight_remaining,
        dataset_name,
        device
):
    g_model_ip_size, g_model_op_size = ca_utils.get_g_model_ip_op_size(
        test_loader,
        device,
        bb_model,
        bb_model_meta,
        torch_concept_vector,
        bb_layer,
        cav_flattening_type,
        dataset_name
    )
    t = Flatten_LR(ip_size=2048, op_size=108).to(device)
    checkpoint_t = os.path.join(cav_path, f"adaptive_avg_pooled_train_model_prune_iteration_{prune_ite}.pth.tar")
    t.load_state_dict(torch.load(checkpoint_t))
    t.eval()
    print(torch_concept_vector[0])
    torch_concept_vector = t.model[0].weight.detach()
    print(f"Cav details: ")
    print(torch_concept_vector.size())
    print(torch_concept_vector[0])

    g = G(g_model_ip_size, g_model_op_size, hidden_features).to(device)
    g.load_state_dict(torch.load(g_model_checkpoint))
    bb_model_mid, bb_model_tail = ca_utils.dissect_bb_model(model_arch, bb_model)

    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_bb = torch.FloatTensor().cuda()
    out_put_predict_g = torch.FloatTensor().cuda()

    g.eval()
    bb_model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_id, (activations, input_to_pred, labels) in enumerate(test_loader):
                bs = activations.size(0)
                activations = activations.to(device)
                input_to_pred = input_to_pred.to(device)
                labels = labels.to(torch.long).to(device)
                norm_vc = ca_utils.get_normalized_vc(
                    activations,
                    torch_concept_vector,
                    th,
                    val_after_th,
                    cav_flattening_type
                )
                concept_to_act = g(norm_vc)
                y_hat_g = ca_utils.get_concept_to_pred(
                    concept_to_act,
                    bs,
                    activations,
                    bb_model_mid,
                    bb_model_tail
                )

                out_put_predict_bb = torch.cat((out_put_predict_bb, input_to_pred), dim=0)
                out_put_GT = torch.cat((out_put_GT, labels), dim=0)
                out_put_predict_g = torch.cat((out_put_predict_g, y_hat_g), dim=0)

                t.set_postfix(batch_id='{0}'.format(batch_id))
                t.update()

    bb_acc = utils.get_correct(out_put_predict_bb, out_put_GT, num_labels) / out_put_GT.size(0)
    g_acc = utils.get_correct(out_put_predict_g, out_put_GT, num_labels) / out_put_GT.size(0)
    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_predict_bb_np = out_put_predict_bb.argmax(dim=1).cpu().numpy()
    out_put_predict_g_np = out_put_predict_g.argmax(dim=1).cpu().numpy()
    completeness_score = utils.cal_completeness_score(num_labels, g_acc, bb_acc)
    metric = {
        "BB": {
            "Accuracy": bb_acc,
            "F1_score": utils.cal_f1_score(out_put_GT_np, out_put_predict_bb_np, avg="macro")
        },
        "G": {
            "Accuracy": g_acc,
            "F1_score": utils.cal_f1_score(out_put_GT_np, out_put_predict_g_np, avg="macro")
        },
        "Completeness_score": completeness_score,
        "percent_weight_remaining": percent_weight_remaining
    }

    return {
        "out_put_GT_np": out_put_GT_np,
        "out_put_predict_bb_np": out_put_predict_bb_np,
        "out_put_predict_g": out_put_predict_g_np,
        "metric": metric,
        "g_acc": g_acc,
        "completeness_score": completeness_score
    }
