from pathlib import Path

def get_config():
    """
    Get the configuration.
    """
    config = {
        "src_lang": "en",
        "tgt_lang": "ne",
        "seq_len": 512,
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "d_model": 512,
        "model_dir": "weights",
        "model_basename": "tmodel_",
        "preload": None, 
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }
    return config

def get_weights_file_path(config, epoch):
    """
    Get the path to the weights file for the given epoch.
    """
    model_dir = config["model_dir"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".")/model_dir/model_filename)