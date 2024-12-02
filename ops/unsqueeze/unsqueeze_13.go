package unsqueeze

import (
	"sort"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinUnsqueeze13Inputs = 2
	MaxUnsqueeze13Inputs = 2
)

// Unsqueeze13 represents the ONNX unsqueeze operator.
type Unsqueeze13 struct{}

// newUnsqueeze13 creates a new unsqueeze operator.
func newUnsqueeze13() ops.Operator {
	return &Unsqueeze13{}
}

// Init initializes the unsqueeze operator.
func (u *Unsqueeze13) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the unsqueeze operator.
func (u *Unsqueeze13) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	dataShape := inputs[0].Shape()

	axes, err := ops.AnyToIntSlice(inputs[1].Data())
	if err != nil {
		return nil, err
	}

	outputRank := len(dataShape) + len(axes)

	if !ops.AllInRange(axes, -outputRank, outputRank-1) {
		return nil, ops.ErrNotAllAxesInRange(outputRank, outputRank)
	}

	// negative entries should be offset by the rank of the output tensor
	// i.e. -1 -> outputRank - 1, -outputrank -> 0
	ops.OffsetArrayIfNegative(axes, outputRank)

	sort.Ints(axes)

	if ops.HasDuplicates(axes) {
		return nil, ops.ErrInvalidInput("axes cannot have duplicate entries after offset", u)
	}

	newShape := insertOnes(dataShape, axes)

	out, ok := inputs[0].Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", inputs[0].Clone())
	}

	err = out.Reshape(newShape...)

	return []tensor.Tensor{out}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (u *Unsqueeze13) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(u, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (u *Unsqueeze13) GetMinInputs() int {
	return MinUnsqueeze13Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (u *Unsqueeze13) GetMaxInputs() int {
	return MaxUnsqueeze13Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (u *Unsqueeze13) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (u *Unsqueeze13) String() string {
	return "unsqueeze13 operator"
}

// Creates a new array, which is `original` with ones added at the indices specified by `indices`
// `indices` may not contain duplicates, the elements are assumed to be in the range 0 <= x < N
// and should be sorted in increasing order.
// Is done in a single pass through the new array with length: len(original) + len(indices).
func insertOnes(original, indices []int) []int {
	N := len(indices) + len(original)

	// Pre-allocate the output shape
	newShape := make([]int, N)

	originalIdx := 0
	indicesIdx := 0

	for i := 0; i < N; i++ {
		if indicesIdx < len(indices) && indices[indicesIdx] == i {
			newShape[i] = 1
			indicesIdx++
		} else {
			newShape[i] = original[originalIdx]
			originalIdx++
		}
	}

	return newShape
}
