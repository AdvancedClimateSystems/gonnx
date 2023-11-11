package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinShapeInputs = 1
	MaxShapeInputs = 1
)

// Shape represents the ONNX shape operator.
type Shape struct{}

// newShape creates a new shape operator.
func newShape() ops.Operator {
	return &Shape{}
}

// Init initializes the shape operator.
func (s *Shape) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply the shape operator to the graph. It creates a node that holds the shape of the
// input node as 1D int64 tensor.
func (s *Shape) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	nodeShape := inputs[0].Shape()
	shape := make([]int64, len(nodeShape))

	for i, dimSize := range nodeShape {
		shape[i] = int64(dimSize)
	}

	out := tensor.New(tensor.WithShape(len(nodeShape)), tensor.WithBacking(shape))

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Shape) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Shape) GetMinInputs() int {
	return MinShapeInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Shape) GetMaxInputs() int {
	return MaxShapeInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Shape) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Shape) String() string {
	return "shape operator"
}
