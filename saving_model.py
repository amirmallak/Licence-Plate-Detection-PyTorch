import torch
import config


def save_model(model):
    # Saving our model
    model_path = config.model_path
    torch.save(model.state_dict(), model_path)
