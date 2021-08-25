from .baseline_dataset import BaselineDataset

def create_dataset(name,file_path,**kwargs):
    if name.lower() == "baseline":
        dataset = BaselineDataset(file_path,**kwargs)
    else:
        dataset = BaselineDataset(file_path,**kwargs)
    return dataset
