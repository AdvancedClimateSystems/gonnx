package slice

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinSlice1Attributes = 2
	MaxSlice1Attributes = 3
)

// Slice1 represents the ONNX slice operator.
type Slice1 struct {
	ops.BaseOperator

	axes   []int
	ends   []int
	starts []int
}

// newSlice1 creates a new slice operator.
func newSlice1() ops.Operator {
	return &Slice1{
		BaseOperator: ops.NewBaseOperator(
			1,
			MinSliceInputs,
			MaxSliceInputs,
			sliceTypeConstraints,
			"slice",
		),
	}
}

// Init initializes the slice operator.
func (s *Slice1) Init(n *onnx.NodeProto) error {
	nAttrs := len(n.GetAttribute())
	if nAttrs < MinSlice1Attributes || nAttrs > MaxSlice1Attributes {
		return ops.ErrInvalidOptionalAttributeCount(MinSlice1Attributes, MaxSlice1Attributes, nAttrs, s)
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
		axes = getDefaultAxes(len(s.starts))
	}

	steps := make([]int, len(s.starts))
	for i := range steps {
		steps[i] = 1
	}

	slices := constructSlices(s.starts, s.ends, steps, axes, len(data.Shape()))

	out, err := data.Slice(slices...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out.Materialize()}, nil
}
