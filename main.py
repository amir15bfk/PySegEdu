import multiprocessing
multiprocessing.freeze_support()
from experiments.experiment_runner import SegmentationExperiment
import torch.multiprocessing as mp
from utils import download
from models import fcn,duck_net, unet,fcbformer,doubleunet,fcn2
import torch

download.download()

experiment = SegmentationExperiment(
    exp_name = "128 100ep",
    dataset = "B",
    model = unet.Unet(),
    load = False,
    model_source = "Trained_models/Unet_B_128 100ep_best.pt",
    root="./data",
    size = (128,128),
    epochs=100,
    batch_size=8,
    num_workers = 0,
    lr=1e-4,#1-4
    lrs=True,#true
    lrs_min=1e-6,
    mgpu=False,
    seed = 42
    )
experiment.run_experiment()
# experiment.report()

