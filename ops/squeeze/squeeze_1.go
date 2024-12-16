package squeeze

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Squeeze1 represents the ONNX squeeze operator.
type Squeeze1 struct {
	ops.BaseOperator

	axes []int
}

// newSqueeze1 creates a new squeeze operator.
func newSqueeze1() ops.Operator {
	return &Squeeze1{
		BaseOperator: ops.NewBaseOperator(
			1,
			1,
			1,
			[][]tensor.Dtype{ops.AllTypes},
			"squeeze",
		),
	}
}

// Init initializes the squeeze operator.
func (s *Squeeze1) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "axes":
			axes, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}

			s.axes = axes
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), s)
		}
	}

	return nil
}

// Apply applies the squeeze operator.
func (s *Squeeze1) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	var err error

	currentShape := inputs[0].Shape()
	nDims := len(currentShape)
	dimsToSqueeze := getDimsToSqueezeFromShape(currentShape)

	if !ops.AllInRange(dimsToSqueeze, 0, nDims-1) {
		return nil, ops.ErrNotAllAxesInRange(nDims, nDims)
	}

	if len(s.axes) > 0 {
		dimsToSqueeze = getDimsToSqueezeFromList(s.axes, nDims)
	}

	newShape := getNewShape(currentShape, dimsToSqueeze)

	out, ok := inputs[0].Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", inputs[0].Clone())
	}

	err = out.Reshape(newShape...)

	return []tensor.Tensor{out}, err
}
