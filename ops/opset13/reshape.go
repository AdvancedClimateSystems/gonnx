package opset13

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// Reshape represents the ONNX reshape operator.
type Reshape struct{}

// newReshape creates a new reshape operator.
func newReshape() ops.Operator {
	return &Reshape{}
}

// Init initializes the reshape operator.
func (r *Reshape) Init(attributes []*onnx.AttributeProto) error {
	return nil
}

// Apply applies the reshape operator.
func (r *Reshape) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	t := inputs[0]
	newShape, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[1].Data().([]int64)))
	if err != nil {
		return nil, err
	}

	err = processShape(newShape, t.Shape())
	if err != nil {
		return nil, err
	}

	out := t.Clone().(tensor.Tensor)
	err = out.Reshape(newShape...)
	return []tensor.Tensor{out}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (r *Reshape) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(r, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (r *Reshape) GetMinInputs() int {
	return 2
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (r *Reshape) GetMaxInputs() int {
	return 2
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (r *Reshape) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (r *Reshape) String() string {
	return "reshape operator"
}

func processShape(newShape, currentShape []int) error {
	for i := 0; i < len(newShape); i++ {
		if newShape[i] == 0 {
			if i >= len(currentShape) {
				return fmt.Errorf("could not infer dim size")
			}
			newShape[i] = currentShape[i]
		}
	}

	// Calculate the total number of elements in the original tensor.
	totalSize := ops.NElements(currentShape...)

	for i := 0; i < len(newShape); i++ {
		// When encountering a -1 dim size, calculate which size this should be.
		if newShape[i] == -1 {
			remainingSize := totalSize
			for j := 0; j < len(newShape); j++ {
				if j == i {
					continue
				}

				if newShape[j] == -1 {
					return fmt.Errorf("At most one -1 dim size is allowed")
				}

				remainingSize /= newShape[j]
			}

			newShape[i] = remainingSize
			break
		}
	}

	return nil
}
