import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from DenseNet121 import DenseNet121
from Utils import Utils
from dataset_classifier import ChestXRay_Classifier


class Tester:
    def __init__(self,
                 labels,
                 image_dir_path,
                 image_source_dir,
                 train_csv_file_name,
                 val_csv_file_name,
                 image_col_header,
                 pre_trained,
                 batch_size,
                 resize,
                 center_crop_size,
                 chk_pt_path,
                 prediction_path,
                 tensor_board_path,
                 channels="RGB",
                 output_size=14,
                 uncertain=1,
                 dataset="",
                 device="cuda:0"):
        self.labels = labels
        self.writer = SummaryWriter(tensor_board_path)
        self.device = device
        self.prediction_path = prediction_path

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(center_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])

        datasets = {
            'val': ChestXRay_Classifier(
                dataset=dataset,
                image_dir_path=image_dir_path,
                image_source_dir=image_source_dir,
                csv_file_name=val_csv_file_name,
                image_col_header=image_col_header,
                transform=transform,
                channels=channels,
                uncertain=uncertain
            )
        }

        self.data_loaders = {
            'val': torch.utils.data.DataLoader(
                datasets['val'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
        }

        self.output_size = output_size
        self.model = DenseNet121(in_channels=channels,
                                 output_size=output_size,
                                 pre_trained=pre_trained).to(self.device)

        model_chk_pt = torch.load(chk_pt_path)
        self.model.load_state_dict(model_chk_pt["state_dict"])

    def predict(self):
        print("<<<==========================================>>>")
        print("Testing is starting...")
        print("<<<==========================================>>>")

        out_put_GT_test, \
        out_put_predict_test, \
        image_name_arr_test, \
        out_prob_arr_test, \
        auroc_mean_test, \
        out_AUROC_test = self._predict_data()

        out_put_GT_test_np = out_put_GT_test.cpu().numpy()
        out_put_predict_test_np = out_put_predict_test.cpu().numpy()
        image_names_test_np = np.asarray(image_name_arr_test)

        np.save(self.prediction_path + "_out_put_GT_test_np_normalize.npy", out_put_GT_test_np)
        np.save(self.prediction_path + "_out_put_predict_test_np_normalize.npy", out_put_predict_test_np)
        np.save(self.prediction_path + "_image_names_test_np_normalize.npy", image_names_test_np)

        return {
            "out_put_GT_test": out_put_GT_test,
            "out_put_predict_test": out_put_predict_test,
            "image_names_test": image_names_test_np,
            "out_prob_arr_test": out_prob_arr_test,
            "auroc_mean_test": auroc_mean_test,
            "out_AUROC_test": out_AUROC_test
        }

    def _predict_data(self):
        self.model.eval()
        image_name_arr = []
        out_put_GT = torch.FloatTensor().cuda()
        out_put_predict = torch.FloatTensor().cuda()
        out_prob_arr = []
        with torch.no_grad():
            with tqdm(total=len(self.data_loaders["val"])) as t:
                for batch_id, (
                        images,
                        labels,
                        image_name
                ) in enumerate(self.data_loaders["val"]):
                    images = images.to(self.device, dtype=torch.float)
                    labels = labels.to(self.device)
                    image_name = image_name[0]
                    out_prob = self.model(images)
                    out_prob_arr.append(out_prob)
                    out_put_predict = torch.cat((out_put_predict, out_prob), dim=0)
                    out_put_GT = torch.cat((out_put_GT, labels), dim=0)
                    image_name_arr.append(image_name)
                    t.set_postfix(batch_id='{0}'.format(batch_id))
                    t.update()

        # print(len(out_prob_arr))
        # print(len(out_prob_arr[0]))
        # print(out_prob_arr[0])
        print(out_put_predict.size())
        print(out_put_GT.size())

        out_AUROC, out_recall = Utils.compute_AUROC_recall(
            out_put_GT,
            out_put_predict,
            self.output_size
        )

        auroc_mean = np.array(out_AUROC).mean()
        print("<<< Model Test Results: AUROC >>>")
        print("MEAN", ": {:.4f}".format(auroc_mean))

        for i in range(0, len(out_AUROC)):
            print(self.labels[i], ': {:.4f}'.format(out_AUROC[i]))
        print("------------------------")

        print("<<< Model Test Results: Recall >>>")

        for i in range(0, len(out_recall)):
            print(self.labels[i], ': {:.4f}'.format(out_recall[i]))
        print("------------------------")

        # out_F1 = Utils.compute_F1(out_put_GT, out_put_predict, self.output_size)
        # F1_mean = np.array(out_F1).mean()
        # print("<<< Model Test Results: F1 >>>")
        # print("MEAN", ": {:.4f}".format(F1_mean))
        #
        # for i in range(0, len(out_F1)):
        #     print(i)
        #     print(self.labels[i], ': {:.4f}'.format(out_F1[i]))
        # print("")
        # print(out_put_GT.device)
        print("#######################################")
        return out_put_GT, out_put_predict, image_name_arr, out_prob_arr, auroc_mean, out_AUROC
