package shape

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinShape1Inputs = 1
	MaxShape1Inputs = 1
)

// Shape1 represents the ONNX shape operator.
type Shape1 struct{}

// newShape1 creates a new shape operator.
func newShape1() ops.Operator {
	return &Shape1{}
}

// Init initializes the shape operator.
func (s *Shape1) Init(*onnx.NodeProto) error {
	return nil
}

// Apply the shape operator to the graph. It creates a node that holds the shape of the
// input node as 1D int64 tensor.
func (s *Shape1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	nodeShape1 := inputs[0].Shape()
	shape := make([]int64, len(nodeShape1))

	for i, dimSize := range nodeShape1 {
		shape[i] = int64(dimSize)
	}

	out := tensor.New(tensor.WithShape(len(nodeShape1)), tensor.WithBacking(shape))

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Shape1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Shape1) GetMinInputs() int {
	return MinShape1Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Shape1) GetMaxInputs() int {
	return MaxShape1Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Shape1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Shape1) String() string {
	return "shape1 operator"
}
