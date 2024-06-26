import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import multiprocessing
multiprocessing.freeze_support()
from experiments.experiment_runner import SegmentationExperiment
from models import fcn,duck_net, unet,fcbformer,doubleunet


model = unet.Unet()

experiment = SegmentationExperiment(
    exp_name = "352 100ep",
    dataset = "B",
    model = model,
    load = False,
    model_source = "Trained_models/FCBFormer_B_352 100ep_last.pt",
    root="./data",
    size = (64,64),
    epochs=5,
    batch_size=4,
    num_workers = 0,
    lr=1e-4,#1-4
    lrs=True,#true
    lrs_min=1e-6,
    mgpu=False,
    seed = 42,
    full_loss=True
    )
experiment.run_experiment()
experiment.report(plot=True)

