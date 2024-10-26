from models.train_module import TrainModule
from utils import initialize_logging, load_config

config = load_config(config_path="./config/local_config.yaml")
initialize_logging(config_path="./config/logging_config.yaml", debug=False)
with open('wandb_token.txt', 'r') as f:
    wandb_token = f.read().strip()
trainer = TrainModule(config, wandb_token, config_path="./config/config.yaml")
trainer.pipeline()
#trainer.test()
