package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Slice represents the ONNX slice operator.
type Slice struct{}

// newSlice creates a new slice operator.
func newSlice() ops.Operator {
	return &Slice{}
}

// Init initializes the slice operator.
func (s *Slice) Init(attributes []*onnx.AttributeProto) error {
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

	axes := s.getDefaultAxes(len(starts))
	if inputs[3] != nil {
		axes, err = ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[3].Data()))
		if err != nil {
			return nil, err
		}
	}

	steps := s.getDefaultSteps(len(starts))
	if inputs[4] != nil {
		steps, err = ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[4].Data()))
		if err != nil {
			return nil, err
		}
	}

	slices := s.constructSlices(starts, ends, steps, axes, len(data.Shape()))

	out, err := data.Slice(slices...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out.Materialize()}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Slice) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Slice) GetMinInputs() int {
	return 3
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Slice) GetMaxInputs() int {
	return 5
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Slice) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		ops.AllTypes,
		{tensor.Int32, tensor.Int64},
		{tensor.Int32, tensor.Int64},
		{tensor.Int32, tensor.Int64},
		{tensor.Int32, tensor.Int64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Slice) String() string {
	return "slice operator"
}

// constructSlice constructs a list with tensor.Slice objects. The list is initializes with nils.
// The axes parameter determines at which indices tensor.Slice objects are placed.
func (s *Slice) constructSlices(starts, ends, steps, axes []int, nTotalSlices int) []tensor.Slice {
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
func (s *Slice) getDefaultAxes(nSlices int) []int {
	axes := make([]int, nSlices)
	for i := 0; i < nSlices; i++ {
		axes[i] = i
	}
	return axes
}

// getDefaultSteps returns the default steps data. By default the steps are 1.
func (s *Slice) getDefaultSteps(nSlices int) []int {
	steps := make([]int, nSlices)
	for i := 0; i < nSlices; i++ {
		steps[i] = 1
	}
	return steps
}
