package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinSqueezeInputs = 1
	MaxSqueezeInputs = 2
)

// Squeeze represents the ONNX squeeze operator.
type Squeeze struct{}

// newSqueeze creates a new squeeze operator.
func newSqueeze() ops.Operator {
	return &Squeeze{}
}

// Init initializes the squeeze operator.
func (s *Squeeze) Init(_ []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the squeeze operator.
func (s *Squeeze) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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

	if inputs[1] != nil {
		dimsToSqueeze, err = getDimsToSqueezeFromTensor(inputs[1], nDims)
		if err != nil {
			return nil, err
		}
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
func (s *Squeeze) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(s, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (s *Squeeze) GetMinInputs() int {
	return MinSqueezeInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (s *Squeeze) GetMaxInputs() int {
	return MaxSqueezeInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (s *Squeeze) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (s *Squeeze) String() string {
	return "squeeze operator"
}

// getDimsToSqueezeFromTensor creates a list with ints representing the dimensions/axes to squeeze
// based on a tensor. The tensor should contain dimensions/axes to squeeze. Negative dimensions
// represent dimensions counting from the end of the shape, i.e. -2 repesents the second
// last dimension.
func getDimsToSqueezeFromTensor(t tensor.Tensor, nDims int) ([]int, error) {
	dimsToSqueeze, err := ops.AnyToIntSlice(t.Data())
	if err != nil {
		return nil, err
	}

	for i, val := range dimsToSqueeze {
		if val < 0 {
			dimsToSqueeze[i] = nDims + val
		}
	}

	return dimsToSqueeze, nil
}

// getDimsToSqueezeFromShape creates a list with ints representing the dimensions/axes to squeeze
// based on the current shape. All dimensions with only 1 value will be squeezed.
func getDimsToSqueezeFromShape(shape []int) []int {
	result := []int{}

	for i, size := range shape {
		if size == 1 {
			result = append(result, i)
		}
	}

	return result
}

// getNewShape returns a new shape based on the current shape and a list of dims to squeeze.
func getNewShape(currentShape tensor.Shape, dimsToSqueeze []int) []int {
	newShape := []int{}

	for i, dimSize := range currentShape {
		if keepDim(i, dimsToSqueeze) {
			newShape = append(newShape, dimSize)
		}
	}

	return newShape
}

// keepDim determines whether or not a dimension should be kept.
func keepDim(dim int, dimsToSqueeze []int) bool {
	for _, squeezeDim := range dimsToSqueeze {
		if dim == squeezeDim {
			return false
		}
	}

	return true
}
