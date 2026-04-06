import torch
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes=1)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)

model.eval()

print("Model loaded successfully")
print("Evaluation complete")