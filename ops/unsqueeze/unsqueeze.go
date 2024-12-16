package unsqueeze

import (
	"sort"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var unsqueezeTypeConstraints = [][]tensor.Dtype{
	ops.AllTypes,
	{tensor.Int64},
}

const (
	MinUnsqueezeInputs = 2
	MaxUnsqueezeInputs = 2
)

// Unsqueeze represents the ONNX unsqueeze operator.
type Unsqueeze struct {
	ops.BaseOperator
}

// newUnsqueeze creates a new unsqueeze operator.
func newUnsqueeze(version int, typeConstraint [][]tensor.Dtype) ops.Operator {
	return &Unsqueeze{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinUnsqueezeInputs,
			MaxUnsqueezeInputs,
			typeConstraint,
			"unsqueeze",
		),
	}
}

// Init initializes the unsqueeze operator.
func (u *Unsqueeze) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the unsqueeze operator.
func (u *Unsqueeze) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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
		return nil, ops.ErrInvalidInput("axes cannot have duplicate entries after offset", u.BaseOperator)
	}

	newShape := insertOnes(dataShape, axes)

	out, ok := inputs[0].Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", inputs[0].Clone())
	}

	err = out.Reshape(newShape...)

	return []tensor.Tensor{out}, err
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
