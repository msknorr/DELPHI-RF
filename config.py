import sys
import torch

# experiment_v3: paper version 25.01.23

class TrainGlobalConfig:
    device = torch.device("cuda")
    num_folds = 5
    num_workers = 8
    use_quantized = False  # if True, call trace_models first (see main.py) & set device="cpu"
    persistent_workers = False if num_workers == 0 else True
    n_epochs = 30
    lr = 0.0005
    batch_size = 16
    verbose = True
    verbose_step = 1
    experiment_name = "/experiment_v3/"
    folder = "D:/weights/" + experiment_name if sys.platform.startswith('win') else "./weights/" + experiment_name
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = False
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=3,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=0.00001,
        eps=1e-08
    )

    external = False
    if external:
        use_quantized = True
        device = torch.device("cpu")
        folder = "../" + experiment_name

config = TrainGlobalConfig()