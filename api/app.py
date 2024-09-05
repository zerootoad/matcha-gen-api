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

normal_model_x = LinearRegressionModel()
normal_model_y = LinearRegressionModel()

rage_model_x = LinearRegressionModel()
rage_model_y = LinearRegressionModel()

normal_model_x.load_state_dict(torch.load("./models/normal/best_model_x.pth", map_location=torch.device('cpu')))
normal_model_y.load_state_dict(torch.load("./models/normal/best_model_y.pth", map_location=torch.device('cpu')))
rage_model_x.load_state_dict(torch.load("./models/rage/best_model_x.pth", map_location=torch.device('cpu')))
rage_model_y.load_state_dict(torch.load("./models/rage/best_model_y.pth", map_location=torch.device('cpu')))

normal_model_x.eval()
normal_model_y.eval()

rage_model_x.eval()
rage_model_y.eval()

@app.route('/predict', methods=['GET'])
def predict():
    ping = request.args.get('ping', type=int)
    model = request.args.get('model', type=int, default=0)
    if ping is None:
        return jsonify({"error": "ping is required"}), 400
    if model is None:
        return jsonify({"error": "model is required (0: rage, 1: normal)"}), 400

    ping_tensor = torch.tensor([ping], dtype=torch.float32).unsqueeze(dim=1)
    
    with torch.no_grad():
        if model == 0:
            pred_x = normal_model_x(ping_tensor).item()
            pred_y = normal_model_y(ping_tensor).item()
        elif model == 1:
            pred_x = rage_model_x(ping_tensor).item()
            pred_y = rage_model_y(ping_tensor).item()
        else:
            return jsonify({"error": "model not existant (0: rage, 1: normal)"}), 400

    return jsonify({"ping": ping, "x": pred_x, "y": pred_y})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
