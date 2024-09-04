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

model_x = LinearRegressionModel()
model_y = LinearRegressionModel()

model_x.load_state_dict(torch.load("./models/best_model_x.pth", map_location=torch.device('cpu')))
model_y.load_state_dict(torch.load("./models/best_model_y.pth", map_location=torch.device('cpu')))

model_x.eval()
model_y.eval()

@app.route('/predict', methods=['GET'])
def predict():
    ping = request.args.get('ping', type=float)
    if ping is None:
        return jsonify({"error": "ping is required"}), 400

    ping_tensor = torch.tensor([ping], dtype=torch.float32).unsqueeze(dim=1)
    
    with torch.no_grad():
        pred_x = model_x(ping_tensor).item()
        pred_y = model_y(ping_tensor).item()

    return jsonify({"ping": ping, "x": pred_x, "y": pred_y})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
