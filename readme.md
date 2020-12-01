
# Getting started
## Installation
```
git clone https://tfs.university.innopolis.ru/tfs/ai_lab/Medicine/_git/segmentation_pipeline
cd segmentation_pipeline
pip install -r requirements.txt
pre-commit install
```


## Train model:
```
python train.py configs/config.yml
```
When it is runs first time, you need to provide access token of telegram bot. <br> 
Go to `@BotFather` and create your own bot.

## Add own dataset:
To implement your own dataset with name `custom` you need to create `data/custom_dataset.py` and implement class `CustomDataset` that inherits from `BaseDataset` and implements all corresponding methods. 
After that you need to put path and name of dataset into config file.

# Documentation 

## Configuration file

Let's go for each section of config file:

### `defaults`
Contains random seed of the experiment.
### `description`
A small description to characterize the experiment.
### `datasets`
##### `steps_per_epoch` - How many steps will one epoch last.
If you use multiple datasets then this number will be multiplied by number of train datasets.

<span style="color:red">**Warning**</span>: If you set `steps_per_epoch` a half of the actual amount of dataset steps, then model will be trained only on the first part of data. 
When using multiple datasets, set `steps_per_epoch` at least to cover biggest dataset. 

##### `list: [dataset]` - list of all project datasets.

##### `dataset` - defines one dataset.

+ `id` - unique id of dataset that used for mapping specific head, loss.
+ `name` - name of dataset to load dataloader implementation. For example: `name='human'` will load `HumanDataset` from `data/human_dataset.py`.
+ `root` - path to the data on your system.
+ `use_to_train` - will model learn from this dataset.
+ `use_to_validate` - will model validate on this dataset. Currently pipeline supports only 1 validation dataset because of [this](https://github.com/PyTorchLightning/pytorch-lightning/issues/3994) issue.
+ `pre_transforms` - Applied to `train`/`valid`/`test` data samples.
+ `augmentations` - Applied to `train` data samples.
+ `post_transforms` - Applied to `train`/`valid`/`test` data samples.

`pre_transforms`, `augmentations`, `post_transforms` - are lists of albumentations transforms. Firstly, code searches for transform in `data/helpers/transforms.py`. If code doesn't find it, then it loads transform from albumentations. 

For several train datasets you can set different image size, because samples from each dataset will be processed in separate batches.

If you have N train datasets, then training will be like:
+ Train on batch from 1st dataset.
+ Train on batch from 2nd dataset.
+ ...
+ Train on batch from Nth dataset.
+ Train on batch from 1st dataset.
+ ...

### `dataloader`
Arguments that will be passed into  `torch.utils.data.DataLoader`

### `model`
Config to create model. May be dependent on task (classification, segmentation, etc.)

This documentation considers segmentation task.

**Warning**: Number of classes always must be >=2. In binary segmentation case it is equal 2. 

##### `name` - name of model
##### `n_dim` - number of dimensions of model and input data.
##### `pipeline` - which pipeline to load from `pipelines/`
##### `params` - dict of parameters that will be passed into model. For segmentation task it will be passed into child class of EncoderDecoder. `params` are specific to model's `name`.
##### `classification`/`segmentation` - defines project-specific modules.
##### `default_criterion` - Default criterion for some task.
##### `heads: [head]` - list of model's heads.

##### `head` - Defines head module to predict specific target.
+ `datasets_ids: [dataset.id]` - list of dataset's id's that will be mapped to this head.
+ `target` - name of target that this head will predict. Dataset realization must return dict with `target` key. All sample's keys will try to map to heads by this `target`.
+ `criterion` - criterion for this head. Fist it searches criterion in `models/criterions.py`. If not finds, then loads from dot-path. Example: `pytorch_toolbelt.losses.SoftCrossEntropyLoss`.
+ `metrics: [metric]` - list of metrics that will be tracked on validation for this head. Metric name in pipeline will be `{target}@{metric.name}`
+ `head` - name of head module. These heads modules are defined in `models/modules.py`.
+ `params` - parameters of head module.

### `optimizer`
List of optimizers. For simple tasks like classification/segmentation there is only one optimizer.

##### `cls` - dot-like path to optimizer. Example: `torch.optim.Adam`.
##### `lr` - learning rate.
##### `Other parameters` - optimizer-specific.

### `scheduler`
List of schedulers. For simple tasks like classification/segmentation there is only one scheduler.

By default, scheduler is called at the end of each epoch. **TODO: add parameters that determines when scheduler is called.** 

##### `cls` - dot-like path to scheduler. Example: `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`.
##### `Other parameters` - scheduler-specific.

### `callbacks`
List of pytorch-lightning callback. Custom callback can be added in `pipelines/callbacks.py`

Supported callbacks:
+ `ModelCheckpoint`
+ `EarlyStopping`
+ `GPUStatsMonitor`
+ `GradientAccumulationScheduler`
+ `LearningRateMonitor`

### `logging`
Which logger to use.

##### `name` - name of logger. 
Available:
+ `csv_logs`
+ `mlflow`
+ `neptune`
+ `comet`
+ `tensorboard`
+ `test_tube`
+ `swandb`

##### `params` - parameters of logger.

### `lightning`
Put here `pytorch_lightning.Trainer` parameters. For example: `gpus`, `num_nodes`.

### `train_stages: [stage]`
Training is divided into stages where one stage is different from another by different components. For example you can change dataset parameters, augmentations, pre and post processing, loss function, optimizer, scheduler, batch_size and others. 

##### `name` - name of stage.

What you can change:
+ `lightning` - set `max_epochs` or change other parameters defined in config above.
+ `datasets.list` - change dataset's parameters. Dataset is identified by id. Here you can change `use_to_train`, `use_to_validate`, `pre_transforms`, `augmentations`, `post_transforms`.
+ `optimizer` - set new optimizer.
+ `scheduler` - set new scheduler.
+ `model.{task}.heads[target_name].criterion` - change criterion for some head. You need to specify `task` and `target` to identify head.
