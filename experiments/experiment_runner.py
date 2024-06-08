import torch

import time
import numpy as np
import glob
import os

import torch
import torch.nn as nn
from datasets import dataloaders
from utils import metrics
from utils import losses


# class SegmentationExperiment:
#     def __init__(self, dataset, root="./data", epochs=200, batch_size=16,num_workers=2, lr=1e-4, lrs=True, lrs_min=1e-6, mgpu=False):
#         self.dataset = dataset
#         self.root = root
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.lr = lr
#         self.lrs = lrs
#         self.lrs_min = lrs_min
#         self.mgpu = mgpu

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.train_dataloader, _, self.val_dataloader = self._get_dataloaders()
#         self.Dice_loss = losses.SoftDiceLoss()
#         self.BCE_loss = nn.BCELoss()
#         self.perf = metrics.DiceScore()
#         self.model = self._initialize_model()
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

#     def _get_dataloaders(self):
#         if self.dataset == "Kvasir":
#             img_path = os.path.join(self.root, "Kvasir-SEG/images/*")
#             depth_path = os.path.join(self.root, "Kvasir-SEG/masks/*")
#         elif self.dataset == "CVC":
#             img_path = os.path.join(self.root, "Original/*")
#             depth_path = os.path.join(self.root, "Ground Truth/*")
        
#         input_paths = sorted(glob.glob(img_path))
#         target_paths = sorted(glob.glob(depth_path))
        
#         return dataloaders.get_dataloaders(input_paths, target_paths, batch_size=self.batch_size,num_workers= self.num_workers)

#     def _initialize_model(self):
#         model = fcbformer.FCBFormer()
#         if self.mgpu:
#             model = nn.DataParallel(model)
#         model.to(self.device)
#         return model

#     def train_epoch(self,epoch):
#         self.model.train()
#         loss_accumulator = []
#         for batch_idx, (data, target) in enumerate(self.train_dataloader):
#             data, target = data.to(self.device), target.to(self.device)
#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.Dice_loss(output, target) + self.BCE_loss(torch.sigmoid(output), target)
#             loss.backward()
#             self.optimizer.step()
#             loss_accumulator.append(loss.item())
#             print(f"\rTrain Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(self.train_dataloader.dataset)} "
#                   f"({100.0 * (batch_idx + 1) / len(self.train_dataloader):.1f}%)]\tLoss: {loss.item():.6f}\t", end="")
#         print(f"\rTrain Epoch: {epoch} [{len(self.train_dataloader.dataset)}/{len(self.train_dataloader.dataset)} "
#               f"(100.0%)]\tAverage loss: {np.mean(loss_accumulator):.6f}\t")
#         return np.mean(loss_accumulator)

#     @torch.no_grad()
#     def test(self, epoch):
#         self.model.eval()
#         perf_accumulator = []
#         for batch_idx, (data, target) in enumerate(self.val_dataloader):
#             data, target = data.to(self.device), target.to(self.device)
#             output = self.model(data)
#             perf_accumulator.append(self.perf(output, target).item())
#             print(f"\rTest  Epoch: {epoch} [{batch_idx + 1}/{len(self.val_dataloader)} "
#                   f"({100.0 * (batch_idx + 1) / len(self.val_dataloader):.1f}%)]\tAverage performance: "
#                   f"{np.mean(perf_accumulator):.6f}\t", end="")
#         print(f"\rTest  Epoch: {epoch} [{len(self.val_dataloader)}/{len(self.val_dataloader)} "
#               f"(100.0%)]\tAverage performance: {np.mean(perf_accumulator):.6f}\t")
#         return np.mean(perf_accumulator), np.std(perf_accumulator)

#     def run_experiment(self):
#         if not os.path.exists("./Trained_models"):
#             os.makedirs("./Trained_models")

#         prev_best_test = None
#         if self.lrs:
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 self.optimizer, mode="max", factor=0.5, min_lr=self.lrs_min, verbose=True
#             )

#         for epoch in range(1, self.epochs + 1):
#             try:
#                 loss = self.train_epoch(epoch)
#                 test_measure_mean, test_measure_std = self.test(epoch)
#             except KeyboardInterrupt:
#                 print("Training interrupted by user")
#                 break

#             if self.lrs:
#                 scheduler.step(test_measure_mean)

#             if prev_best_test is None or test_measure_mean > prev_best_test:
#                 print("Saving...")
#                 torch.save(
#                     {
#                         "epoch": epoch,
#                         "model_state_dict": self.model.state_dict() if not self.mgpu else self.model.module.state_dict(),
#                         "optimizer_state_dict": self.optimizer.state_dict(),
#                         "loss": loss,
#                         "test_measure_mean": test_measure_mean,
#                         "test_measure_std": test_measure_std,
#                     },
#                     f"Trained_models/FCBFormer_{self.dataset}.pt",
#                 )
#                 prev_best_test = test_measure_mean



class SegmentationExperiment:
    def __init__(self, 
                dataset, model,
                exp_name = "",
                load = False,
                model_source = None,
                root = './data',
                size = (352,352),
                epochs = 200, 
                batch_size = 16, 
                lr = 1e-4, 
                lrs = True, 
                lrs_min = 1e-6, 
                mgpu = False, 
                num_workers = 4,
                seed = 42):
        self.dataset = dataset
        self.exp_name = exp_name
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lrs = lrs
        self.lrs_min = lrs_min
        self.mgpu = mgpu
        self.num_workers = num_workers
        self.model = model
        self.load = load
        self.size = size
        self.seed = seed
        if self.load:
            checkpoint = torch.load(model_source)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataloader, _, self.val_dataloader = self._get_dataloaders()
        self.Dice_loss = losses.SoftDiceLoss()
        self.BCE_loss = nn.BCELoss()
        self.perf = metrics.DiceScore()
        self.model = self._initialize_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def _get_dataloaders(self):
        if self.dataset == "Kvasir":
            img_path = os.path.join(self.root, "Kvasir-SEG/images/*")
            depth_path = os.path.join(self.root, "Kvasir-SEG/masks/*")
            input_paths = sorted(glob.glob(img_path))
            target_paths = sorted(glob.glob(depth_path))
        elif self.dataset == "CVC":
            img_path = os.path.join(self.root, "CVC-ClinicDB/Original/*")
            depth_path = os.path.join(self.root, "CVC-ClinicDB/Ground Truth/*")
            input_paths = sorted(glob.glob(img_path))
            target_paths = sorted(glob.glob(depth_path))
        elif self.dataset == "B":
            img_path1 = os.path.join(self.root, "Kvasir-SEG/images/*")
            depth_path1 = os.path.join(self.root, "Kvasir-SEG/masks/*")
            img_path2 = os.path.join(self.root, "CVC-ClinicDB/Original/*")
            depth_path2 = os.path.join(self.root, "CVC-ClinicDB/Ground Truth/*")
            input_paths = sorted(glob.glob(img_path1))+sorted(glob.glob(img_path2))
            target_paths = sorted(glob.glob(depth_path1))+ sorted(glob.glob(depth_path2))
        
        return dataloaders.get_dataloaders(input_paths, target_paths, batch_size=self.batch_size, num_workers=self.num_workers,input_dims = self.size,seed = self.seed)

    def _initialize_model(self):
        if self.mgpu:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        return self.model

    def train_epoch(self, epoch):
        self.model.train()
        loss_accumulator = []
        t = time.time()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.Dice_loss(output, target) #+ self.BCE_loss(torch.sigmoid(output), target)
            loss.backward()
            self.optimizer.step()
            loss_accumulator.append(loss.item())
            print(f"\rTrain Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(self.train_dataloader.dataset)} "
                  f"({100.0 * (batch_idx + 1) / len(self.train_dataloader):.1f}%)]\tLoss: {loss.item():.6f}\t"
                  f"Time: {time.time() - t:.6f}", end="")
        print(f"\rTrain Epoch: {epoch} [{len(self.train_dataloader.dataset)}/{len(self.train_dataloader.dataset)} "
              f"(100.0%)]\tAverage loss: {np.mean(loss_accumulator):.6f}\tTime: {time.time() - t:.6f}")
        return np.mean(loss_accumulator)

    @torch.no_grad()
    def test(self, epoch):
        self.model.eval()
        perf_accumulator = []
        t = time.time()
        for batch_idx, (data, target) in enumerate(self.val_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            perf_accumulator.append(self.perf(output, target).item())
            print(f"\rTest  Epoch: {epoch} [{batch_idx + 1}/{len(self.val_dataloader)} "
                  f"({100.0 * (batch_idx + 1) / len(self.val_dataloader):.1f}%)]\tAverage performance: "
                  f"{np.mean(perf_accumulator):.6f}\tTime: {time.time() - t:.6f}", end="")
        print(f"\rTest  Epoch: {epoch} [{len(self.val_dataloader)}/{len(self.val_dataloader)} "
              f"(100.0%)]\tAverage performance: {np.mean(perf_accumulator):.6f}\tTime: {time.time() - t:.6f}")
        return np.mean(perf_accumulator), np.std(perf_accumulator)
    


    def run_test(self):
        self.test(0)

    def run_experiment(self):
        if not os.path.exists("./Trained_models"):
            os.makedirs("./Trained_models")

        prev_best_test = None
        if self.lrs:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, min_lr=self.lrs_min, verbose=True
            )

        for epoch in range(1, self.epochs + 1):
            try:
                loss = self.train_epoch(epoch)
                test_measure_mean, test_measure_std = self.test(epoch)
            except KeyboardInterrupt:
                print("Training interrupted by user")
                break

            if self.lrs:
                scheduler.step(test_measure_mean)

            if prev_best_test is None or test_measure_mean > prev_best_test:
                print("Saving best...")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict() if not self.mgpu else self.model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": loss,
                        "test_measure_mean": test_measure_mean,
                        "test_measure_std": test_measure_std,
                    },
                    f"Trained_models/{self.model.name}_{self.dataset}_{self.exp_name}_best.pt",
                )
                prev_best_test = test_measure_mean
            print("Saving last...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict() if not self.mgpu else self.model.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                f"Trained_models/{self.model.name}_{self.dataset}_{self.exp_name}_last.pt",
            )