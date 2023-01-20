import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 3
HIDDEN_SIZE = 5
OUTPUT_SIZE = 2


class MLP(nn.Module):
    """
    A simple MultiLayer Perceptron (MLP). It has structure:

    MLP(
      (layer1): Linear(in_features=3, out_features=5, bias=True)
      (relu): ReLU()
      (layer2): Linear(in_features=5, out_features=2, bias=True)
    )
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, inputs):
        x = self.relu(self.layer1(inputs))
        return self.layer2(x)


def generate_mlp_onnx_from_torch():
    print("-" * 100)
    print("Generating 'mlp.onnx'...")

    torch.manual_seed(42)
    mlp = MLP()
    print(mlp, "\n")

    batch_size = 2
    sample_in = torch.from_numpy(np.random.rand(batch_size, INPUT_SIZE)).float()

    torch.onnx.export(
        mlp,
        sample_in,
        "./onnx_models/mlp.onnx",
        opset_version=13,
        input_names=["data_input"],
        output_names=["preds"],
        dynamic_axes={
            "data_input": {0: "batch_size"},
            "preds": {0: "batch_size"},
        },
    )


def test_mlp_torch():
    torch.manual_seed(42)
    mlp = MLP()

    batch_size = 2
    data_input = torch.from_numpy(
        np.arange(0, batch_size * INPUT_SIZE).reshape(batch_size, INPUT_SIZE)
    ).float()

    with torch.no_grad():
        preds = mlp(data_input)

    print("-" * 50)
    print("MLP sample:\n")
    print("---INPUTS---")
    print("data_input: ", data_input.shape)
    print(data_input, "\n")
    print("---OUTPUTS---")
    print("preds: ", preds.shape)
    print(preds, "\n")
