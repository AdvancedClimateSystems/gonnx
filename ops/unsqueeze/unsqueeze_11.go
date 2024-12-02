package unsqueeze

import (
	"sort"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinUnsqueeze11Inputs = 2
	MaxUnsqueeze11Inputs = 2
)

// Unsqueeze11 represents the ONNX unsqueeze operator.
type Unsqueeze11 struct {
	axes []int
}

// newUnsqueeze11 creates a new unsqueeze operator.
func newUnsqueeze11() ops.Operator {
	return &Unsqueeze11{}
}

// Init initializes the unsqueeze operator.
func (u *Unsqueeze11) Init(n *onnx.NodeProto) error {
	attrs := n.GetAttribute()
	if len(attrs) != 1 {
		return ops.ErrInvalidAttributeCount(1, len(attrs), u)
	}

	axes, err := ops.AnyToIntSlice(attrs[0].GetInts())
	if err != nil {
		return err
	}

	u.axes = axes

	return nil
}

// Apply applies the unsqueeze operator.
func (u *Unsqueeze11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	dataShape := inputs[0].Shape()

	outputRank := len(dataShape) + len(u.axes)

	if !ops.AllInRange(u.axes, -outputRank, outputRank-1) {
		return nil, ops.ErrNotAllAxesInRange(outputRank, outputRank)
	}

	// negative entries should be offset by the rank of the output tensor
	// i.e. -1 -> outputRank - 1, -outputrank -> 0
	ops.OffsetArrayIfNegative(u.axes, outputRank)

	sort.Ints(u.axes)

	if ops.HasDuplicates(u.axes) {
		return nil, ops.ErrInvalidInput("axes cannot have duplicate entries after offset", u)
	}

	newShape := insertOnes(dataShape, u.axes)

	out, ok := inputs[0].Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", inputs[0].Clone())
	}

	err := out.Reshape(newShape...)

	return []tensor.Tensor{out}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (u *Unsqueeze11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(u, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (u *Unsqueeze11) GetMinInputs() int {
	return MinUnsqueeze11Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (u *Unsqueeze11) GetMaxInputs() int {
	return MaxUnsqueeze11Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (u *Unsqueeze11) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (u *Unsqueeze11) String() string {
	return "unsqueeze11 operator"
}
