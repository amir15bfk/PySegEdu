import multiprocessing
multiprocessing.freeze_support()
from experiments.experiment_runner import SegmentationExperiment
import torch.multiprocessing as mp
from utils import download
from models import fcn,duck_net, unet,fcbformer,doubleunet,fcn2
import torch


download.download()

experiment = SegmentationExperiment(
    exp_name = "352 50ep",
    dataset = "B",
    model = unet.Unet(),
    load = False,
    model_source = "Trained_models/DoubleUnet_B_352 50ep_best.pt",
    root="./data",
    size = (352,352),#(128,128)
    epochs=50,
    batch_size=4,
    num_workers = 0,
    lr=1e-4,#1-4
    lrs=True,#true
    lrs_min=1e-6,
    mgpu=False,
    seed = 42
    )
experiment.run_experiment()
# experiment.report()

