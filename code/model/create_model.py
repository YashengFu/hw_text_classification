from .baselineBert import BaselineBertConfig,BaselineBertModel

def create_model(model_name="baseline",**kwargs):
    """create model

    Args:
        model_name : name of model, default baseline
    Returns:
        model
    """
    if model_name.lower() == "baseline":
        config = BaselineBertConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = BaselineBertModel(config)
        print("Create Baseline Bert Model")
    else:
        config = BaselineBertConfig()
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = BaselineBertModel(config)
        print("Create default(Baseline Bert) Model")
    print(config.__dict__)
    print(model)
    return model
