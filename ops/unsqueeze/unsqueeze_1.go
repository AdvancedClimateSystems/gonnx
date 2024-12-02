package unsqueeze

import (
	"sort"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinUnsqueeze1Inputs = 2
	MaxUnsqueeze1Inputs = 2
)

// Unsqueeze1 represents the ONNX unsqueeze operator.
type Unsqueeze1 struct {
	axes []int
}

// newUnsqueeze1 creates a new unsqueeze operator.
func newUnsqueeze1() ops.Operator {
	return &Unsqueeze1{}
}

// Init initializes the unsqueeze operator.
func (u *Unsqueeze1) Init(n *onnx.NodeProto) error {
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
func (u *Unsqueeze1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
func (u *Unsqueeze1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(u, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (u *Unsqueeze1) GetMinInputs() int {
	return MinUnsqueeze1Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (u *Unsqueeze1) GetMaxInputs() int {
	return MaxUnsqueeze1Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (u *Unsqueeze1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (u *Unsqueeze1) String() string {
	return "unsqueeze1 operator"
}
