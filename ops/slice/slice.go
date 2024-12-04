package slice

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var sliceTypeConstraints = [][]tensor.Dtype{
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
func newSlice(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (s *Slice) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	data := inputs[0]

	starts, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[1].Data()))
	if err != nil {
		return nil, err
	}

	ends, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[2].Data()))
	if err != nil {
		return nil, err
	}

	axes := getDefaultAxes(len(starts))
	if inputs[3] != nil {
		axes, err = ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[3].Data()))
		if err != nil {
			return nil, err
		}
	}

	steps := getDefaultSteps(len(starts))
	if inputs[4] != nil {
		steps, err = ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[4].Data()))
		if err != nil {
			return nil, err
		}
	}

	slices := constructSlices(starts, ends, steps, axes, len(data.Shape()))

	out, err := data.Slice(slices...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out.Materialize()}, nil
}

// constructSlice constructs a list with tensor.Slice objects. The list is initializes with nils.
// The axes parameter determines at which indices tensor.Slice objects are placed.
func constructSlices(starts, ends, steps, axes []int, nTotalSlices int) []tensor.Slice {
	slices := make([]tensor.Slice, nTotalSlices)
	for i := 0; i < nTotalSlices; i++ {
		slices[i] = nil
	}

	for i, ax := range axes {
		if ax < 0 {
			ax = nTotalSlices + ax
		}

		slices[ax] = ops.NewSlicer(starts[i], ends[i], steps[i])
	}

	return slices
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
