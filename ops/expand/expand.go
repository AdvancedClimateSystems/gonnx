package expand

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var expandTypeConstraints = [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}

// Expand represents the ONNX expand operator.
type Expand struct {
	ops.BaseOperator
}

// newExpand creates a new expand operator.
func newExpand(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Expand{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"expand",
		),
	}
}

// Init initializes the expand operator.
func (f *Expand) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the expand operator.
func (f *Expand) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]

	shape, err := ops.AnyToIntSlice(inputs[1].Data())
	if err != nil {
		return nil, err
	}

	// If the new shape has more dimensions than the input tensor, we
	// need to prepend some dimensions to the input tensor shape.
	if len(shape) > len(input.Shape()) {
		input, err = ops.AddExtraDimsToTensor(input, len(shape)-len(input.Shape()))
		if err != nil {
			return nil, err
		}
	}

	for axis := len(shape) - 1; axis >= 0; axis-- {
		if input.Shape()[axis] != shape[axis] {
			input, err = tensor.Repeat(input, axis, shape[axis])
			if err != nil {
				return nil, err
			}
		}
	}

	return []tensor.Tensor{input}, nil
}
