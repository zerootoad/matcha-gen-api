import torch
from torch import nn
from flask import Flask, request, jsonify

app = Flask(__name__)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# Load models
model_x = LinearRegressionModel()
model_y = LinearRegressionModel()

model_x.load_state_dict(torch.load("./models/best_model_x.pth"))
model_y.load_state_dict(torch.load("./models/best_model_y.pth"))

model_x.eval()
model_y.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'ping_value' not in data:
        return jsonify({"error": "ping_value is required"}), 400
    
    ping_value = data['ping_value']
    ping_tensor = torch.tensor([ping_value], dtype=torch.float32).unsqueeze(dim=1)
    
    with torch.no_grad():
        pred_x = model_x(ping_tensor).item()
        pred_y = model_y(ping_tensor).item()

    return jsonify({"ping_value": ping_value, "x": pred_x, "y": pred_y})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
