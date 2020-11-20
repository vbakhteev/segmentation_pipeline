
## Installation
```
git clone https://tfs.university.innopolis.ru/tfs/ai_lab/Medicine/_git/segmentation_pipeline
cd segmentation_pipeline
pip install -r requirements.txt
pre-commit install
```


#### Train model:
```
python train.py configs/config.yml
```
When it is runs first time, you need to provide access token of telegram bot. <br> 
Go to `@BotFather` and create your own bot.

#### Add own dataset:
To implement your own dataset with name `custom` you need to create `data/custom_dataset.py` and implement class `CustomDataset` that inherits from `BaseDataset` and implements all corresponding methods. 
After that you need to put path and name of dataset into config file. 

#### Pipelines:
To add new pipeline (for example, classification) you need to create new file file `pipelines/classifier.py` and implement class `Classifier` that defines training behaviour of models. After that in `pipelines/__init__.py` add name2pipeline mapping in `available_pipelines`.  

#### Transforms and Augmentations:
They are defined in `config.dataset.augmentations` for each experiment. All transforms consist of `pre_transforms, padding and post_transforms` in the same order. `augmentations` apply only to training samples. <br>
Each transformation is based on an albumentations library. The code first looks for a transform class in data `data/transforms.py` and, if not found, imports from albumentations.

#### Metrics:
No stable version yet. We will migrate to new version of `pytorch-lightning` when they will have stable version.

#### Loss functions
Criterion that defined in `config.criterion` can be custom criterion defined in `models/criterions.py` or can be called from any package in internet.
