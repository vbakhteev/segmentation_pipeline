
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

