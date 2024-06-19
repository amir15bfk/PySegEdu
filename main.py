import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import multiprocessing
multiprocessing.freeze_support()
from experiments.experiment_runner import SegmentationExperiment
from utils import download
from models import fcn,duck_net, unet,fcbformer,doubleunet,fcn2


download.download()

experiment = SegmentationExperiment(
    exp_name = "352 100ep",
    dataset = "B",
    model = fcbformer.FCBFormer(),
    load = False,
    model_source = "Trained_models/DoubleUnet_B_352 100ep_last.pt",
    root="./data",
    size = (352,352),
    epochs=100,
    batch_size=6,
    num_workers = 0,
    lr=1e-4,#1-4
    lrs=True,#true
    lrs_min=1e-6,
    mgpu=False,
    seed = 42
    )
experiment.run_experiment()
experiment.report(plot=True)

