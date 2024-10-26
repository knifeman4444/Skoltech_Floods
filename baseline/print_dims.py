from models.custom_models import CustomVIT
model = CustomVIT(in_channels=10)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)