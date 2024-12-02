package shape

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinShape13Inputs = 1
	MaxShape13Inputs = 1
)

// Shape13 represents the ONNX shape operator.
type Shape13 struct{}

// newShape13 creates a new shape operator.
func newShape13() ops.Operator {
	return &Shape13{}
}

// Init initializes the shape operator.
func (s *Shape13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply the shape operator to the graph. It creates a node that holds the shape of the
// input node as 1D int64 tensor.
func (s *Shape13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	nodeShape13 := inputs[0].Shape()
	shape := make([]int64, len(nodeShape13))

	for i, dimSize := range nodeShape13 {
		shape[i] = int64(dimSize)
	}

	out := tensor.New(tensor.WithShape(len(nodeShape13)), tensor.WithBacking(shape))

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Shape13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Shape13) GetMinInputs() int {
	return MinShape13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Shape13) GetMaxInputs() int {
	return MaxShape13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Shape13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Shape13) String() string {
	return "shape13 operator"
}
