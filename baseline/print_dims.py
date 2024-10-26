from models.custom_models import CustomVIT
from utils import load_config

config = load_config(config_path="./config/config.yaml")
model = CustomVIT(config=config)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
