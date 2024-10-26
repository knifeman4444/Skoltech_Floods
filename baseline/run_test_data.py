from models.train_module import TrainModule
from utils import initialize_logging, load_config


config = load_config(config_path="./config/local_config.yaml")
initialize_logging(config_path="./config/logging_config.yaml", debug=False)

models_paths_and_names = [
    ("baseline/outputs_val/run-1/model/best-model-epoch=21.pt", "-"),
    ("baseline/outputs_val/run-1/model/best-model-epoch=21.pt", "-")
]

if len(models_paths_and_names) == 0:
    if not config['test']['model_ckpt']:
        print('Model is not defined!')
    else:
        trainer = TrainModule(config, wandb_token=None, config_path="./config/local_config.yaml")
        trainer.test()
else:
    for model_path, model_name in models_paths_and_names:
        config['test']['model_ckpt'] = model_path
        config['test']['model_name'] = model_name
        trainer = TrainModule(config, wandb_token=None, config_path="./config/local_config.yaml")
        trainer.test()
