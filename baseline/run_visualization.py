from models.train_module import load_model
from models.visualize import visualize_model_predictions
from utils import load_config
import torch

if __name__ == "__main__":
    config = load_config(config_path="./config/config.yaml")
    model = load_model(config).to('cuda')
    model.load_state_dict(torch.load("outputs_val/run-29/model/best-model-epoch=1.pt"), strict=False)
    visualize_model_predictions(model, config)
