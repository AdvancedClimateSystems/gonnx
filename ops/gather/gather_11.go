package gather

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinGather11Inputs = 2
	MaxGather11Inputs = 2
)

// Gather11 represents the ONNX gather operator.
type Gather11 struct {
	axis int // axis to gather on, default is 0
}

// newGather11 creates a new gather operator.
func newGather11() ops.Operator {
	return &Gather11{
		axis: 0,
	}
}

// Init initializes the gather operator.
func (g *Gather11) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) == 1 {
		attr := attributes[0]

		if attr.GetName() == "axis" {
			g.axis = int(attr.GetI())
		} else {
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	} else if len(attributes) > 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), g)
	}

	return nil
}

// Apply applies the gather operator.
func (g *Gather11) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	// Convert the indices (of Dtype Int32 or Int64) to a tensor with Dtype Int
	indicesData, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[1].Data()))
	if err != nil {
		return nil, err
	}

	indices := tensor.New(tensor.WithBacking(indicesData), tensor.WithShape(inputs[1].Shape()...))

	data := inputs[0]

	// Make sure axis is in the correct range (according to the size of the data tensor)
	rank := len(data.Shape())
	dataAxis := g.axis

	if dataAxis < -rank || dataAxis > rank-1 {
		return nil, ops.ErrAxisOutOfRange(rank, rank, dataAxis)
	}
	// Offset axis if a negative index is given.
	if dataAxis < 0 {
		dataAxis += rank
	}

	// Make sure the input indices are all in the correct range (according to the size of the
	// dimension which is selected by `axis`)
	axisDimSize := data.Shape()[dataAxis]
	if !ops.AllInRange(indicesData, -axisDimSize, axisDimSize-1) {
		return nil, ops.ErrNotAllAxesInRange(axisDimSize, axisDimSize)
	}

	err = ops.OffsetTensorIfNegative(indices, axisDimSize)
	if err != nil {
		return nil, err
	}

	// Make the shape of the output tensor
	os := insertWithReplace(indices.Shape(), data.Shape(), dataAxis)
	output := tensor.New(tensor.WithShape(os...), tensor.Of(data.Dtype()))

	// Perform the actual gather operation
	err = gather(output, data, indices, dataAxis)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{output}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (g *Gather11) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(g, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (g *Gather11) GetMinInputs() int {
	return MinGather11Inputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (g *Gather11) GetMaxInputs() int {
	return MaxGather11Inputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (g *Gather11) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		ops.AllTypes,
		{tensor.Int32, tensor.Int64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (g *Gather11) String() string {
	return "gather11 operator"
}
