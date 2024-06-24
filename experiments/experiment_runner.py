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
from utils import visualizations
import matplotlib.pyplot as plt



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
                seed = 42,
                full_loss=True):
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
        self.full_loss = full_loss
        self.model_source = model_source
        if self.load:
            checkpoint = torch.load(model_source)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataloader, self.test_kvasir_dataloader, self.test_cvc_dataloader, self.val_dataloader = self._get_dataloaders()
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
            input_paths = sorted(glob.glob(img_path1) + glob.glob(img_path2))
            target_paths = sorted(glob.glob(depth_path1) + glob.glob(depth_path2))
        
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
            if self.full_loss:
                loss = self.Dice_loss(output, target)  + self.BCE_loss(torch.sigmoid(output), target)
            else:
                loss = self.Dice_loss(output, target)
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
    def val(self, epoch):
        self.model.eval()
        perf_accumulator = []
        t = time.time()
        for batch_idx, (data, target) in enumerate(self.val_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            perf_accumulator.append(self.perf(output, target).item())
            print(f"\rval  Epoch: {epoch} [{batch_idx + 1}/{len(self.val_dataloader)} "
                  f"({100.0 * (batch_idx + 1) / len(self.val_dataloader):.1f}%)]\tAverage performance: "
                  f"{np.mean(perf_accumulator):.6f}\tTime: {time.time() - t:.6f}", end="")
        print(f"\rval  Epoch: {epoch} [{len(self.val_dataloader)}/{len(self.val_dataloader)} "
              f"(100.0%)]\tAverage performance: {np.mean(perf_accumulator):.6f}\tTime: {time.time() - t:.6f}")
        return np.mean(perf_accumulator), np.std(perf_accumulator)
    

    def test_on_dataset(self,dataset,dataloader,metrics=None):
        metrics = [self.perf] if metrics==None else metrics
        perf_accumulator = dict()
        for m in metrics:
            perf_accumulator[m.name]=[]
        t = time.time()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            for m in metrics:
                perf_accumulator[m.name].append(m(output, target).item())
            print(f"\rTest on {dataset} [{batch_idx + 1}/{len(dataloader)} "
                f"({100.0 * (batch_idx + 1) / len(dataloader):.1f}%)] \tTime: {time.time() - t:.6f}", end="")
        print(f"\rTest on {dataset} [{len(dataloader)}/{len(dataloader)}(100%)]\tTime: {time.time() - t:.6f}")
        print("results :")
        out = []
        for m in metrics:
            score = np.mean(perf_accumulator[m.name])*100
            out.append(score)
            print(f" {m.name} : {score:.4f} %")
        return (dataset,out)

    @torch.no_grad()
    def test(self,metrics=None):
        self.model.eval()
        out = []
        if self.dataset=="B":
            out.append(self.test_on_dataset("Kvasir",self.test_kvasir_dataloader,metrics=metrics))
            
            out.append(self.test_on_dataset("CVC",self.test_cvc_dataloader,metrics=metrics))
        elif self.dataset=="CVC":
            out.append(self.test_on_dataset("CVC",self.test_cvc_dataloader,metrics=metrics))
        elif self.dataset=="Kvasir":
            out.append(self.test_on_dataset("Kvasir",self.test_kvasir_dataloader,metrics=metrics))
        return out
    

    def run_test(self):
        self.test()
    
    def report(self,plot=False,metrics=[metrics.PrecisionScore(),metrics.RecallScore(),metrics.F1Score(),metrics.DiceScore(),metrics.mIoUScore()]):
        print(self.dataset)
        tests = self.test(metrics=metrics)
        data = {"Metrics":[i.name for i in metrics]}
        for (k,v) in tests:
            data[k] = v
        if plot:
            visualizations.plot_metrics(data,self.model.name)
    
    def plot_predictions(self, num_samples=1, output_dir='out'):
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.model.eval()
        dataloader = self.test_cvc_dataloader

        for idx, (data, target) in enumerate(dataloader):
            if idx >= num_samples:
                break

            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)

            data = data.cpu().numpy().transpose(0, 2, 3, 1)
            target = target.cpu().numpy().transpose(0, 2, 3, 1)
            output = torch.sigmoid(output).detach().cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(data.shape[0]):
                # Save input image (assuming it's a single-channel grayscale image)
                input_image = data[i, :, :, 0]  # Extract the first channel

                # Ensure input image is in [0, 1] range
                input_image = input_image - np.min(input_image)
                input_image = input_image / np.max(input_image)

                plt.imsave(os.path.join(output_dir, f"sample_{idx * data.shape[0] + i}_input.png"), input_image, cmap='gray')

                # Save ground truth mask
                gt_mask = target[i, :, :, 0]  # Extract the first channel

                # Ensure ground truth mask is in [0, 1] range
                gt_mask = gt_mask - np.min(gt_mask)
                gt_mask = gt_mask / np.max(gt_mask)

                plt.imsave(os.path.join(output_dir, f"sample_{idx * data.shape[0] + i}_ground_truth.png"), gt_mask, cmap='gray')

                # Save predicted mask
                pred_mask = output[i, :, :, 0]  # Extract the first channel

                # Ensure predicted mask is in [0, 1] range
                pred_mask = pred_mask - np.min(pred_mask)
                pred_mask = pred_mask / np.max(pred_mask)

                plt.imsave(os.path.join(output_dir, f"{self.model.name}.png"), pred_mask, cmap='gray')
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
                test_measure_mean, test_measure_std = self.val(epoch)
                self.report()
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