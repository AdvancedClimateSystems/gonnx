import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx

INPUT_SIZE = 3

sample_data = np.array(
    [
        [1, 10, 100],
        [1.5, 13, 120],
        [0.8, 9, 95],
        [0.9, 11, 105],
        [0.6, 12, 101],
        [1.3, 10, 110],
        [1.1, 7, 108],
    ],
    dtype=np.float32,
)


def generate_scaler_onnx_from_sklearn():
    print("-" * 100)
    print("Generating 'scaler.onnx'...")

    scaler = StandardScaler()
    scaler.fit(sample_data)

    scaler = Pipeline(steps=[("scaler", scaler)])

    onnx_scaler = to_onnx(scaler, sample_data, target_opset=13)
    with open("./onnx_models/scaler.onnx", "wb") as onnx_file:
        onnx_file.write(onnx_scaler.SerializeToString())

    print(scaler)


def test_scaler_sklearn():
    scaler = StandardScaler()
    scaler.fit(sample_data)

    Y = scaler.transform(sample_data[:2])

    print("-" * 50)
    print("Scaler sample:\n")
    print("---INPUTS---")
    print("X: ", sample_data[:2].shape)
    print(sample_data[:2], "\n")
    print("---OUTPUTS---")
    print("Y: ", Y.shape)
    print(Y, "\n")
