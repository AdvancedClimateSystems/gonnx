package where

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var whereTypeConstraints = [][]tensor.Dtype{
	{tensor.Bool},
	ops.AllTypes,
	ops.AllTypes,
}

// Where represents the ONNX where operator.
type Where struct {
	ops.BaseOperator
}

// newWhere creates a new where operator.
func newWhere(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Where{
		BaseOperator: ops.NewBaseOperator(
			version,
			3,
			3,
			typeConstraints,
			"where",
		),
	}
}

// Init initializes the where operator.
func (w *Where) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the where operator.
func (w *Where) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	condition := inputs[0]

	X := inputs[1]
	Y := inputs[2]

	X, Y, err := ops.MultidirectionalBroadcast(X, Y)
	if err != nil {
		return nil, err
	}

	condition, X, err = ops.MultidirectionalBroadcast(condition, X)
	if err != nil {
		return nil, err
	}

	out, err := where(X, Y, condition)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, err
}

func where(X, Y, condition tensor.Tensor) (tensor.Tensor, error) {
	out := tensor.New(tensor.Of(X.Dtype()), tensor.WithShape(X.Shape()...))

	iterator := condition.Iterator()
	iterator.Reset()

	for !iterator.Done() {
		coords := iterator.Coord()

		conditionRaw, err := condition.At(coords...)
		if err != nil {
			return nil, err
		}

		conditionValue, ok := conditionRaw.(bool)
		if !ok {
			return nil, ops.ErrCast
		}

		var value any
		if conditionValue {
			value, err = X.At(coords...)
		} else {
			value, err = Y.At(coords...)
		}

		if err != nil {
			return nil, err
		}

		err = out.SetAt(value, coords...)
		if err != nil {
			return nil, err
		}

		_, err = iterator.Next()
		if err != nil {
			return nil, err
		}
	}

	return out, nil
}
