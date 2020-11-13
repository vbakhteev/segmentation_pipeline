from yacs.config import CfgNode as CN


def get_default():
    _C = CN()

    # Defaults settings -----------------------------------------------
    _C.defaults = CN()
    _C.defaults.seed = 0

    # Dataset settings ------------------------------------------------
    _C.dataset = CN()
    _C.dataset.root = "/home/semyon/data/chest_xray"
    _C.dataset.steps_per_epoch = 0
    _C.dataset.pre_transforms = []
    _C.dataset.augmentations = []
    _C.dataset.transforms = []

    # DataLoader
    _C.dataloader = CN()
    _C.dataloader.num_workers = 1
    _C.dataloader.batch_size = 1
    _C.dataloader.pin_memory = True

    # Model settings --------------------------------------------------
    _C.model = CN()
    _C.model.name = "Unet"
    _C.model.pipeline = "segmentation"  # [segmentation]
    _C.model.dim = 2
    _C.model.params = {"encoder_name": "resnet34", "in_channels": 1, "classes": 1}

    # Criterion
    _C.criterion = [{"cls": "pytorch_toolbelt.losses.DiceLoss"}]

    # Optimizer
    _C.optimizer = [{"cls": "torch.optim.Adam", "lr": 0.001}]

    # Scheduler
    _C.scheduler = [{"name": "StepLR", "step_size": 10}]

    # Training settings -----------------------------------------------
    # Checkpointing
    _C.checkpointing = CN()
    _C.checkpointing.save_top_k = 1
    _C.checkpointing.save_last = False
    _C.checkpointing.mode = "max"  # if 'max' then bigger metric is better.
    _C.checkpointing.metric = CN()
    _C.checkpointing.metric.name = "intersection_over_union"
    _C.checkpointing.metric.threshold = 0.5

    # Logging
    _C.logging = CN()
    _C.logging.metrics = [{"name": "intersection_over_union", "threshold": 0.5}]

    # Lightning. You can add any argument for pytorch_lightning.Trainer
    _C.lightning = CN()
    _C.lightning.gpus = 1
    _C.lightning.num_nodes = 1

    # Parameters defined in training stages overwrite parameters defined above
    _C.train_stages = [
        {"pretraining": {"lightning": {"max_epochs": 20}}},
        {
            "fine_tuning": {
                "lightning": {"max_epochs": 20},
                "optimizer": [{"lr": 0.00001}],
            }
        },
    ]
    return _C


def get_configuration(filename):
    """
    Obtain dict-like configuration object based on default configuration and updated
    with specified file.
    :param filename: path to .yaml configs file
    :return: CfgNode object
    """
    cfg = get_default()
    cfg.merge_from_file(filename)

    # cfg.freeze() # Uncomment not to be able modify values with dot notation
    return cfg


if __name__ == "__main__":
    import sys

    _C = get_default()
    if len(sys.argv) == 1:
        sys.argv.append("default.yaml")
    print(_C)
    with open(sys.argv[1], "w") as f:
        print(_C, file=f)
