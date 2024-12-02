package squeeze

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinSqueeze1Inputs = 1
	MaxSqueeze1Inputs = 2
)

// Squeeze1 represents the ONNX squeeze operator.
type Squeeze1 struct {
	axes []int
}

// newSqueeze1 creates a new squeeze operator.
func newSqueeze1() ops.Operator {
	return &Squeeze1{}
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

	if !ops.AllInRange(dimsToSqueeze, -nDims, nDims-1) {
		return nil, ops.ErrNotAllAxesInRange(nDims, nDims)
	}

	// negative entries should be offset by the rank of the output tensor
	// i.e. -1 -> nDims - 1, -nDims -> 0
	ops.OffsetArrayIfNegative(dimsToSqueeze, nDims)

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

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (s *Squeeze1) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Squeeze1) GetMinInputs() int {
	return MinSqueeze1Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Squeeze1) GetMaxInputs() int {
	return MaxSqueeze1Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Squeeze1) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Squeeze1) String() string {
	return "squeeze1 operator"
}
