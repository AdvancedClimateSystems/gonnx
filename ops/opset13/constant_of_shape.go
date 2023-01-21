package opset13

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// ConstantOfShape represents the ONNX constant of shape operator.
type ConstantOfShape struct {
	// One element tensor, giving the value and type of the output tensor
	// defaults to value 0 and type float32.
	value *tensor.Dense
}

// newConstantOfShape creates a new constant of shape operator.
func newConstantOfShape() ops.Operator {
	return &ConstantOfShape{}
}

// Init initializes the constant of shape operator.
func (op *ConstantOfShape) Init(attributes []*onnx.AttributeProto) error {
	if len(attributes) > 1 {
		return fmt.Errorf(ops.InvalidAttrCountErrTemplate, op, "0 or 1", len(attributes))
	}

	if len(attributes) == 1 {
		attr := attributes[0]
		if attr.GetName() == "value" {
			t, err := onnx.TensorFromProto(attr.GetT())
			if err != nil {
				return err
			}

			op.value = tensor.New(tensor.WithBacking(t.Data()))
			if op.value.Len() != 1 {
				return fmt.Errorf(
					"Value input tensor should be a single element tensor, but was %v",
					op.value,
				)
			}
		} else {
			return fmt.Errorf(ops.UnknownAttributeErrTemplate, op, attr.GetName())
		}
	} else {
		// Default
		op.value = tensor.New(tensor.FromScalar(float32(0.0)))
	}

	return nil
}

// Apply applies the constant of shape operator.
func (op *ConstantOfShape) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	shape, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[0].Data()))
	if err != nil {
		return nil, err
	}

	// Empty dimensions in a tensor are not supported
	for i := range shape {
		if shape[i] <= 0 {
			return nil, fmt.Errorf(
				"Non positive dimensions are not allowed (must be > 0). Given: %v",
				shape,
			)
		}
	}
	t := tensor.New(tensor.WithShape(shape...), tensor.Of(op.value.Dtype()))
	t, err = t.AddScalar(op.value, true)

	return []tensor.Tensor{t}, err
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (op *ConstantOfShape) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(op, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (op *ConstantOfShape) GetMinInputs() int {
	return 1
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (op *ConstantOfShape) GetMaxInputs() int {
	return 1
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (op *ConstantOfShape) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Int64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (op *ConstantOfShape) String() string {
	return "constant of shape operator"
}
