import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import multiprocessing
multiprocessing.freeze_support()
from experiments.experiment_runner import SegmentationExperiment
import torch.multiprocessing as mp
from utils import download
from models import fcn,duck_net, unet,fcbformer,doubleunet,fcn2



# download.download()
out = "out/c3"
experiment = SegmentationExperiment(
    exp_name = "352 100ep",
    dataset = "B",
    model = fcn2.FCN8s(),
    load = True,
    model_source = "Trained_models/FCN8s_B_352 50ep_best.pt",
    root="./data",
    size = (352,352),
    epochs=100,
    batch_size=4,
    num_workers = 0,
    lr=1e-4,#1-4
    lrs=True,#true
    lrs_min=1e-6,
    mgpu=False,
    seed = 42
    )
# experiment.run_experiment()
# experiment.report(plot=True)
experiment.plot_predictions(output_dir=out)

experiment = SegmentationExperiment(
    exp_name = "352 100ep",
    dataset = "B",
    model = unet.Unet(),
    load = True,
    model_source = "Trained_models/Unet_B_128 100ep_best.pt",
    root="./data",
    size = (128,128),
    epochs=100,
    batch_size=4,
    num_workers = 0,
    lr=1e-4,#1-4
    lrs=True,#true
    lrs_min=1e-6,
    mgpu=False,
    seed = 42
    )
# experiment.run_experiment()
# experiment.report(plot=True)
experiment.plot_predictions(output_dir=out)

experiment = SegmentationExperiment(
    exp_name = "352 100ep",
    dataset = "B",
    model = doubleunet.build_doubleunet(),
    load = True,
    model_source = "Trained_models/DoubleUnet_B_352 100ep_best4.pt",
    root="./data",
    size = (352,352),
    epochs=100,
    batch_size=4,
    num_workers = 0,
    lr=1e-4,#1-4
    lrs=True,#true
    lrs_min=1e-6,
    mgpu=False,
    seed = 42
    )
# experiment.run_experiment()
# experiment.report(plot=True)
experiment.plot_predictions(output_dir=out)

experiment = SegmentationExperiment(
    exp_name = "352 100ep",
    dataset = "B",
    model = fcbformer.FCBFormer(),
    load = True,
    model_source = "Trained_models/FCBFormer_B_352 20ep_best.pt",
    root="./data",
    size = (352,352),
    epochs=100,
    batch_size=4,
    num_workers = 0,
    lr=1e-4,#1-4
    lrs=True,#true
    lrs_min=1e-6,
    mgpu=False,
    seed = 42
    )
# experiment.run_experiment()
# experiment.report(plot=True)
experiment.plot_predictions(output_dir=out)