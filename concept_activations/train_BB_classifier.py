import datetime
import time

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from DenseNet121 import DenseNet121
from Utils import Utils
from dataset_classifier import ChestXRay_Classifier


class Trainer:
    def __init__(self,
                 image_dir_path,
                 image_source_dir,
                 train_csv_file_name,
                 val_csv_file_name,
                 image_col_header,
                 pre_trained,
                 epochs,
                 batch_size,
                 resize,
                 center_crop_size,
                 chk_pt_path,
                 tensor_board_path,
                 loss_path,
                 channels="RGB",
                 output_size=14,
                 lr=0.0001,
                 betas=(0.9, 0.999),
                 eps=1e-08,
                 weight_decay=0,
                 uncertain=1,
                 dataset="",
                 device="cuda:0"):
        self.epochs = epochs
        self.model_chk_pt_path = chk_pt_path
        self.loss_path = loss_path

        self.writer = SummaryWriter(tensor_board_path)
        self.device = device
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Scale(resize),
                transforms.CenterCrop(center_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Scale(resize),
                transforms.CenterCrop(center_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }

        # transform = transforms.Compose([
        #     transforms.Resize(resize),
        #     transforms.CenterCrop(center_crop_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std),
        # ])

        # transform = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        datasets = {
            'train': ChestXRay_Classifier(
                dataset=dataset,
                image_dir_path=image_dir_path,
                image_source_dir=image_source_dir,
                csv_file_name=train_csv_file_name,
                image_col_header=image_col_header,
                transform=data_transforms["train"],
                channels=channels,
                uncertain=uncertain,
                image_size=resize,
                crop_size=center_crop_size,
                normalize=True
            ),
            'val': ChestXRay_Classifier(
                dataset=dataset,
                image_dir_path=image_dir_path,
                image_source_dir=image_source_dir,
                csv_file_name=val_csv_file_name,
                image_col_header=image_col_header,
                transform=data_transforms["val"],
                channels=channels,
                uncertain=uncertain,
                image_size=resize,
                crop_size=center_crop_size,
                normalize=True
            )
        }

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        self.data_loaders = {
            'train': torch.utils.data.DataLoader(
                datasets['train'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            ),
            'val': torch.utils.data.DataLoader(
                datasets['val'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )
        }

        self.model = DenseNet121(in_channels=channels,
                                 output_size=output_size,
                                 pre_trained=pre_trained).to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr,
                                    betas=betas,
                                    eps=eps,
                                    weight_decay=weight_decay)

    def fit(self):
        train_loss_arr = []
        val_loss_arr = []
        loss_min = 10000
        train_start = []
        train_end = []
        print("<<<==========================================>>>")
        print("Training is starting...")
        print("<<<==========================================>>>")
        for epoch_id in range(0, self.epochs):
            train_start.append(time.time())
            loss_train = self._train(epoch_id)
            train_loss_arr.append(loss_train)

            train_end.append(time.time())
            loss_val = self._val(epoch_id)
            val_loss_arr.append(loss_val)

            Utils.save_model(epoch_id,
                             loss_train,
                             loss_val,
                             loss_min,
                             self.model.state_dict(),
                             self.optimizer.state_dict(),
                             chk_pt_path=self.model_chk_pt_path + "/seq_epoch_{0}_time_{1}.pth.tar"
                             .format(epoch_id,
                                     datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
            print("===================================> Epoch:{0}".format(epoch_id),
                  " Training loss: {:.3f}".format(loss_train),
                  " Valid loss: {:.3f} <===================================".format(loss_val))
            np.save(
                self.loss_path +
                "/seq_epoch_{0}_time_{1}_train_loss.npy".format(
                    epoch_id,
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                ), np.asarray(train_loss_arr))
            np.save(
                self.loss_path +
                "/seq_epoch_{0}_time_{1}_val_loss.npy".format(
                    epoch_id,
                    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                ), np.asarray(val_loss_arr))

            if loss_val < loss_min:
                loss_min = loss_val
                Utils.save_model(epoch_id,
                                 loss_train,
                                 loss_val,
                                 loss_min,
                                 self.model.state_dict(),
                                 self.optimizer.state_dict(),
                                 chk_pt_path=self.model_chk_pt_path + "/best_epoch_{0}_time_{1}.pth.tar"
                                 .format(epoch_id,
                                         datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
                print("[Best loss] ===> Epoch: {0} [save] train loss: {1} valid loss: {2}".
                      format(epoch_id, loss_train, loss_val))
            else:
                print("===> Epoch: {0} [----] valid loss: {1}".format(epoch_id, loss_val))

            # break

        train_time = np.array(train_end) - np.array(train_start)
        print("Training ended. Total time taken: {0}".format(train_time.round(0)))

        np.save(
            self.loss_path +
            "/final_epoch_{0}_time_{1}_train_loss.npy".format(
                self.epochs,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            ), np.asarray(train_loss_arr))
        np.save(
            self.loss_path +
            "/final_epoch_{0}_time_{1}_val_loss.npy".format(
                self.epochs,
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            ), np.asarray(val_loss_arr))

        Utils.save_model(self.epochs,
                         np.mean(np.asarray(train_loss_arr)),
                         np.mean(np.asarray(val_loss_arr)),
                         loss_min,
                         self.model.state_dict(),
                         self.optimizer.state_dict(),
                         chk_pt_path=self.model_chk_pt_path + "/Final_epoch_{0}_time_{1}.pth.tar"
                         .format(self.epochs,
                                 datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

    def _train(self, epoch):
        running_loss = 0
        self.model.train()
        print("#######################################")
        print("Train set!! Epoch: {0}".format(epoch))
        with tqdm(total=len(self.data_loaders["train"])) as t:
            for batch_id, (images, labels, image_name) in enumerate(self.data_loaders["train"]):

                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device)
                pred_labels = self.model(images)
                loss = self.criterion(pred_labels, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if batch_id % 1000 == 999:
                    self.writer.add_scalar("training loss for epoch: {0}".format(epoch),
                                           running_loss / 1000,
                                           epoch * len(self.data_loaders['train']) + batch_id)

                print("batch id: {0}, loss: {1}".format(batch_id, loss))
                t.set_postfix(epoch='{0}'.format(epoch), training_loss='{:05.3f}'.format(running_loss))
                t.update()

                # break
        print("#######################################")
        return running_loss / len(self.data_loaders['train'])

    def _val(self, epoch):
        running_loss = 0
        self.model.eval()
        print("#######################################")
        print("Validation set!! Epoch: {0}".format(epoch))
        with torch.no_grad():
            with tqdm(total=len(self.data_loaders["val"])) as t:
                for batch_id, (images, labels, image_name) in enumerate(self.data_loaders["val"]):
                    images = images.to(self.device, dtype=torch.float)
                    labels = labels.to(self.device)
                    pred_labels = self.model(images)
                    loss = self.criterion(pred_labels, labels)
                    running_loss += loss.item()

                    if batch_id % 1000 == 999:
                        self.writer.add_scalar("training loss for epoch: {0}".format(epoch),
                                               running_loss / 1000,
                                               epoch * len(self.data_loaders['val']) + batch_id)
                    print("batch id: {0}, loss: {1}".format(batch_id, loss))

                    t.set_postfix(epoch='{0}'.format(epoch), validation_loss='{:05.3f}'.format(running_loss))
                    t.update()

                    # break

        print("#######################################")
        return running_loss / len(self.data_loaders['val'])
