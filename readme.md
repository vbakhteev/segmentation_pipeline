
#### Train model:
```
python train.py configs/config.yml
```

#### Add own dataset:
To implement your own dataset with name `custom` you need to create `data/custom_dataset.py` and implement class `CustomDataset` that inherits from `BaseDataset` and implements all corresponding methods. 
After that you need to put path and name of dataset into config file. 

#### Add own model:
