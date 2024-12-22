package cumsum

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var cumsumTypeConstraints = [][]tensor.Dtype{
	{tensor.Int32, tensor.Int64, tensor.Uint32, tensor.Uint64, tensor.Float32, tensor.Float64},
	{tensor.Int32, tensor.Int64},
}

// CumSum represents the ONNX cumsum operator.
type CumSum struct {
	ops.BaseOperator

	exclusive bool
	reverse   bool
}

// newCumSum creates a new cumsum operator.
func newCumSum(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &CumSum{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"cumsum",
		),
		exclusive: false,
		reverse:   false,
	}
}

// Init initializes the cumsum operator.
func (c *CumSum) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "exclusive":
			c.exclusive = attr.GetI() == 1
		case "reverse":
			c.reverse = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), c)
		}
	}

	return nil
}

// Apply applies the cumsum operator.
func (c *CumSum) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	axis, err := ops.AnyToInt(inputs[1].ScalarValue())
	if err != nil {
		return nil, err
	}

	out, err := cumsum(inputs[0], axis, c.exclusive, c.reverse)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// Performs cumulative sum of the input elements along the given axis.
// Exclusive means the the cumsum for position j will not include the j-th element.
// Reverse means the cumsum will be performed in reverse order.
func cumsum(x tensor.Tensor, axis int, exclusive, reverse bool) (tensor.Tensor, error) {
	out, ok := x.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrCast
	}

	nDims := len(x.Shape())
	axis = ops.ConvertNegativeAxis(axis, nDims)

	if axis < 0 || axis >= nDims {
		return nil, ops.ErrAxisOutOfRange(0, nDims, axis)
	}

	axisSize := x.Shape()[axis]

	var startValue int
	if reverse {
		startValue = axisSize - 1
	} else {
		startValue = 0
	}

	slices := make([]tensor.Slice, nDims)
	slices[axis] = ops.NewSlicer(startValue, startValue+1)

	prevView, err := x.Slice(slices...)
	if err != nil {
		return nil, err
	}

	prevValues := prevView.Materialize()

	for i := startValue; endValueReached(i, axisSize, reverse); {
		slices[axis] = ops.NewSlicer(i, i+1)

		currentView, err := out.Slice(slices...)
		if err != nil {
			return nil, err
		}

		currentValues := currentView.Materialize()

		switch {
		// If exclusive is true, the first result in the cumsum opertaion is zero.
		// We can achieve this by subtracting the current values from the current values.
		// This way we don't have to infer the underlying type of the tensor.
		case i == startValue && exclusive:
			zeroValues, err := ops.Sub(currentValues, currentValues)
			if err != nil {
				return nil, err
			}

			err = tensor.Copy(currentView, zeroValues)
			if err != nil {
				return nil, err
			}

		case i != startValue && exclusive:
			err = tensor.Copy(currentView, prevValues)
			if err != nil {
				return nil, err
			}

			newValues, err := ops.Add(currentValues, prevValues)
			if err != nil {
				return nil, err
			}

			prevValues = newValues
		case i != startValue:
			newValues, err := ops.Add(currentValues, prevValues)
			if err != nil {
				return nil, err
			}

			err = tensor.Copy(currentView, newValues)
			if err != nil {
				return nil, err
			}

			prevValues = newValues
		}

		if reverse {
			i--
		} else {
			i++
		}
	}

	return out, nil
}

func endValueReached(i, axisSize int, reverse bool) bool {
	if reverse {
		return i >= 0
	}

	return i < axisSize
}
