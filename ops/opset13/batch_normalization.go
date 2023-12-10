package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinBatchNormalizationInputs = 5
	MaxBatchNormalizationInputs = 5
)

// BatchNormalization represents the ONNX batchNormalization operator.
type BatchNormalization struct {
	epsilon  float32
	momentum float32

	outputs []string
}

// newBatchNormalization creates a new batchNormalization operator.
func newBatchNormalization() ops.Operator {
	return &BatchNormalization{
		epsilon:  1e-5,
		momentum: 0.9,
	}
}

// Init initializes the batchNormalization operator.
func (b *BatchNormalization) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "epsilon":
			b.epsilon = attr.GetF()
		case "momentum":
			b.momentum = attr.GetF()
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), b)
		}
	}

	b.outputs = n.GetOutput()

	return nil
}

// Apply applies the batchNormalization operator.
func (b *BatchNormalization) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	X := inputs[0]
	scale := inputs[1]
	B := inputs[2]
	mean := inputs[3]
	variance := inputs[4]

	return []tensor.Tensor{}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (b *BatchNormalization) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(b, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (b *BatchNormalization) GetMinInputs() int {
	return MinBatchNormalizationInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (b *BatchNormalization) GetMaxInputs() int {
	return MaxBatchNormalizationInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (b *BatchNormalization) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (b *BatchNormalization) String() string {
	return "batchNormalization operator"
}
