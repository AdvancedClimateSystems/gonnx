package reshape

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var reshapeTypeConstraints = [][]tensor.Dtype{ops.AllTypes, {tensor.Int64}}

const (
	ReshapeMinInputs = 2
	ReshapeMaxInputs = 2
)

// Reshape represents the ONNX reshape operator.
type Reshape struct {
	ops.BaseOperator
}

// newReshape creates a new reshape operator.
func newReshape(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Reshape{
		BaseOperator: ops.NewBaseOperator(
			version,
			ReshapeMinInputs,
			ReshapeMaxInputs,
			typeConstraints,
			"reshape",
		),
	}
}

// Init initializes the reshape operator.
func (r *Reshape) Init(*onnx.NodeProto) error {
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

	out, ok := t.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", t.Clone())
	}

	err = out.Reshape(newShape...)

	return []tensor.Tensor{out}, err
}

func processShape(newShape, currentShape []int) error {
	for i := 0; i < len(newShape); i++ {
		if newShape[i] == 0 {
			if i >= len(currentShape) {
				return ops.ErrDimension("could not infer dim size")
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
					return ops.ErrDimension("at most one -1 dim size is allowed")
				}

				remainingSize /= newShape[j]
			}

			newShape[i] = remainingSize

			break
		}
	}

	return nil
}
