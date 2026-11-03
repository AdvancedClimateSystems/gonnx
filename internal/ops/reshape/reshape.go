package reshape

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var reshapeTypeConstraints = [][]tensor.DataType{ops.AllTypes, {tensor.Int64}}

const (
	ReshapeMinInputs = 2
	ReshapeMaxInputs = 2
)

// Reshape represents the ONNX reshape operator.
type Reshape struct {
	ops.BaseOperator
}

// newReshape creates a new reshape operator.
func newReshape(version int, typeConstraints [][]tensor.DataType) ops.Operator {
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
func (r *Reshape) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	t := inputs[0]

	newShape, err := ops.AnyToIntSlice(inputs[1].Data())
	if err != nil {
		return nil, err
	}

	err = processShape(newShape, t.Shape())
	if err != nil {
		return nil, err
	}

	out, err := t.Reshape(newShape...)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor{out}, nil
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
