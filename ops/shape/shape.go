package shape

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var shapeTypeConstraints = [][]tensor.Dtype{ops.AllTypes}

// Shape represents the ONNX shape operator.
type Shape struct {
	ops.BaseOperator
}

// newShape creates a new shape operator.
func newShape(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Shape{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"shape",
		),
	}
}

// Init initializes the shape operator.
func (s *Shape) Init(*onnx.NodeProto) error {
	return nil
}

// Apply the shape operator to the graph. It creates a node that holds the shape of the
// input node as 1D int64 tensor.
func (s *Shape) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	nodeShape := inputs[0].Shape()
	shape := make([]int64, len(nodeShape))

	for i, dimSize := range nodeShape {
		shape[i] = int64(dimSize)
	}

	out := tensor.New(tensor.WithShape(len(nodeShape)), tensor.WithBacking(shape))

	return []tensor.Tensor{out}, nil
}
