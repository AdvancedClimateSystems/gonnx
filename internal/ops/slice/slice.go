package slice

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var sliceTypeConstraints = [][]tensor.DataType{
	ops.AllTypes,
	{tensor.Int32, tensor.Int64},
	{tensor.Int32, tensor.Int64},
	{tensor.Int32, tensor.Int64},
	{tensor.Int32, tensor.Int64},
}

const (
	MinSliceInputs = 3
	MaxSliceInputs = 5
)

// Slice represents the ONNX slice operator.
type Slice struct {
	ops.BaseOperator
}

// newSlice creates a new slice operator.
func newSlice(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Slice{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinSliceInputs,
			MaxSliceInputs,
			typeConstraints,
			"slice",
		),
	}
}

// Init initializes the slice operator.
func (s *Slice) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the slice operator.
func (s *Slice) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	data := inputs[0]

	starts, err := ops.AnyToIntSlice(inputs[1].Data())
	if err != nil {
		return nil, err
	}

	ends, err := ops.AnyToIntSlice(inputs[2].Data())
	if err != nil {
		return nil, err
	}

	axes := getDefaultAxes(len(starts))
	if inputs[3] != nil {
		axes, err = ops.AnyToIntSlice(inputs[3].Data())
		if err != nil {
			return nil, err
		}
	}

	steps := getDefaultSteps(len(starts))
	if inputs[4] != nil {
		steps, err = ops.AnyToIntSlice(inputs[4].Data())
		if err != nil {
			return nil, err
		}
	}

	out, err := ops.Slice(data, starts, ends, axes, steps)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
}

// getDefaultAxes returns the default axes parameter. By default the slices are in natural order.
func getDefaultAxes(nSlices int) []int {
	axes := make([]int, nSlices)
	for i := 0; i < nSlices; i++ {
		axes[i] = i
	}

	return axes
}

// getDefaultSteps returns the default steps data. By default the steps are 1.
func getDefaultSteps(nSlices int) []int {
	steps := make([]int, nSlices)
	for i := 0; i < nSlices; i++ {
		steps[i] = 1
	}

	return steps
}
