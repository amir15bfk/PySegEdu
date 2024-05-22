import torch
from datasets import DatasetLoader
from models.unet import UNet
from utils.metrics import iou_score
from utils.visualizations import visualize_results

class ExperimentRunner:
    def __init__(self, dataset_path, model, epochs=20, batch_size=8):
        self.dataset_loader = DatasetLoader(dataset_path)
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        # Training logic
        pass

    def evaluate(self):
        # Evaluation logic
        pass

    def run(self):
        self.train()
        self.evaluate()