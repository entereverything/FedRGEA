import logging
import numpy as np
import copy
from types import SimpleNamespace

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from utils.losses import IntraSCL
from tqdm import tqdm

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    def get_num_class_list(self):
        self.n_classes = self.dataset.n_classes
        class_num = np.array([0] * self.n_classes)
        for idx in self.idxs:
            label = self.dataset.targets[idx]
            class_num[label] += 1
        return class_num.tolist()


class LocalUpdate(object):
    def __init__(self, args, id, dataset, idxs):
        self.args = args
        self.id = id
        self.local_dataset = DatasetSplit(dataset, idxs)
        self.class_num_list = self.local_dataset.get_num_class_list()
        logging.info(
            f"Client{id} ===> Each class num: {self.class_num_list}, Total: {len(self.local_dataset)}")
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr
        # Server-broadcast class prototypes [num_classes, feat_dim]; set each round in train.py
        self.global_protos = None
        # FedFSA: server-broadcast momentum vector (CPU); disrupt layers persist across rounds
        self.fedfsa_client_momentum_vec = None
        self.fedfsa_max2layer = []

    def _local_present_inverse_freq_weights(self, device="cuda"):

        num_per_cls = torch.tensor(self.class_num_list, dtype=torch.float32, device=device)
        present = num_per_cls > 0
        w = torch.zeros_like(num_per_cls)
        w[present] = 1.0 / num_per_cls[present]
        w = w / torch.clamp(w.sum(), min=1e-12) * float(len(self.class_num_list))
        return w
    
    
    def train(self, net, writer):
        net.train()
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        print(f"Id: {self.id}, Num: {len(self.local_dataset)}")

        # train and update
        epoch_loss = []
        ce_criterion = nn.CrossEntropyLoss()
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for (images, labels) in self.ldr_train:
                if isinstance(images, list):
                    images = images[0]
                images, labels = images.cuda(), labels.cuda()

                _, logits = net(images)
                loss = ce_criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

        return net.state_dict(), np.array(epoch_loss).mean()
