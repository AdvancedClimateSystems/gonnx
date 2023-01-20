import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 3
HIDDEN_SIZE = 5


class GRU(nn.Module):
    """
    Simple GRU model with only a single GRU unit. Structure:

    GRU(
      (gru): GRU(3, 5, batch_first=True)
    )
    """

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)

    def forward(self, inputs, hidden):
        x, new_hidden = self.gru(inputs, hidden)
        return x, new_hidden

    def get_init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, HIDDEN_SIZE))


def generate_gru_onnx_from_torch():
    print("-" * 100)
    print("Generating 'gru.onnx'...")

    torch.manual_seed(42)
    gru = GRU()
    print(gru, "\n")

    batch_size = 1
    seq_length = 30
    data_input = torch.from_numpy(
        np.arange(0, batch_size * seq_length * INPUT_SIZE).reshape(
            batch_size, seq_length, INPUT_SIZE
        )
    ).float()

    sample_in = (data_input, gru.get_init_hidden(batch_size))
    torch.onnx.export(
        gru,
        sample_in,
        "./onnx_models/gru.onnx",
        opset_version=13,
        input_names=["data_input", "init_hidden"],
        output_names=["preds", "hidden_out"],
        dynamic_axes={
            "data_input": {0: "batch_size", 1: "seq_length"},
            "init_hidden": {1: "batch_size"},
            "preds": {0: "batch_size", 1: "seq_length"},
            "hidden_out": {1: "batch_size"},
        },
    )


def test_gru_torch():
    torch.manual_seed(42)
    gru = GRU()

    batch_size = 1
    seq_length = 30
    data_input = torch.from_numpy(
        np.arange(0, batch_size * seq_length * INPUT_SIZE).reshape(
            batch_size, seq_length, INPUT_SIZE
        )
    ).float()
    init_hidden = gru.get_init_hidden(batch_size)

    with torch.no_grad():
        preds, hidden_out = gru(data_input, init_hidden)

    print("-" * 50)
    print("GRU sample:\n")
    print("---INPUTS---")
    print("data_input: ", data_input.shape)
    print(data_input, "\n")
    print("init_hidden: ", init_hidden.shape)
    print(init_hidden, "\n")
    print("---OUTPUTS---")
    print("preds: ", preds.shape)
    print(preds, "\n")
    print("hidden_out: ", hidden_out.shape)
    print(hidden_out, "\n")
