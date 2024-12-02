package slice

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinSliceAttributes = 2
	MaxSliceAttributes = 3
	MinSlice1Inputs    = 3
	MaxSlice1Inputs    = 5
)

// Slice1 represents the ONNX slice operator.
type Slice1 struct {
	axes   []int
	ends   []int
	starts []int
}

// newSlice1 creates a new slice operator.
func newSlice1() ops.Operator {
	return &Slice1{}
}

// Init initializes the slice operator.
func (s *Slice1) Init(n *onnx.NodeProto) error {
	nAttrs := len(n.GetAttribute())
	if nAttrs < 2 || nAttrs > 3 {
		return ops.ErrInvalidOptionalAttributeCount(MinSliceAttributes, MaxSliceAttributes, nAttrs, s)
	}

	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "axes":
			axes, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}

			s.axes = axes
		case "ends":
			ends, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}

			s.ends = ends
		case "starts":
			starts, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}

			s.starts = starts
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), s)
		}
	}

	return nil
}

// Apply applies the slice operator.
func (s *Slice1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	data := inputs[0]

	axes := s.axes
	if len(s.axes) == 0 {
		axes = s.getDefaultAxes(len(s.starts))
	}

	slices := s.constructSlices(s.starts, s.ends, axes, len(data.Shape()))

	out, err := data.Slice(slices...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out.Materialize()}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Slice1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Slice1) GetMinInputs() int {
	return MinSlice1Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Slice1) GetMaxInputs() int {
	return MaxSlice1Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Slice1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		ops.AllTypes,
		{tensor.Int32, tensor.Int64},
		{tensor.Int32, tensor.Int64},
		{tensor.Int32, tensor.Int64},
		{tensor.Int32, tensor.Int64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Slice1) String() string {
	return "slice1 operator"
}

// constructSlice constructs a list with tensor.Slice objects. The list is initializes with nils.
// The axes parameter determines at which indices tensor.Slice objects are placed.
func (s *Slice1) constructSlices(starts, ends, axes []int, nTotalSlices int) []tensor.Slice {
	slices := make([]tensor.Slice, nTotalSlices)
	for i := 0; i < nTotalSlices; i++ {
		slices[i] = nil
	}

	for i, ax := range axes {
		if ax < 0 {
			ax = nTotalSlices + ax
		}

		slices[ax] = ops.NewSlicer(starts[i], ends[i])
	}

	return slices
}

// getDefaultAxes returns the default axes parameter. By default the slices are in natural order.
func (s *Slice1) getDefaultAxes(nSlices int) []int {
	axes := make([]int, nSlices)
	for i := 0; i < nSlices; i++ {
		axes[i] = i
	}

	return axes
}
