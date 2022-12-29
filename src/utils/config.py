import yaml
import dataclasses

@dataclasses.dataclass(frozen=True)
class Config:
    root_dir : str
    model_name : str
    batch_size : int
    num_workers : int
    num_sanity_val_steps : int
    input_size : int
    out_features : int
    learning_rate : float
    early_stopping_metric : str
    pretrained : bool
    max_epocks : int
    min_epocks : int
    gpus : list
    output_path : str
    name : str
    wandb : bool
    train : bool
    log_every_n_steps : int
    split : bool
    test : bool
    opt : bool
    semi : bool
    iteration : int
    makevideo : bool
    remove : bool
    video_name : str

    
def get_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)
    return config
    
