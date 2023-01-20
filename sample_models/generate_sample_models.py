import argparse

from python_models import (
    generate_mlp_onnx_from_torch,
    generate_gru_onnx_from_torch,
    generate_scaler_onnx_from_sklearn,
    test_mlp_torch,
    test_gru_torch,
    test_scaler_sklearn,
)


def main(args):
    if args.action == "generate":
        generate_mlp_onnx_from_torch()
        generate_gru_onnx_from_torch()
        generate_scaler_onnx_from_sklearn()
    elif args.action == "test":
        test_mlp_torch()
        test_gru_torch()
        test_scaler_sklearn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        choices=["generate", "test"],
        required=True,
        help="Whether to generate sample models or test sample models (generate output)",
    )
    args = parser.parse_args()
    main(args)
