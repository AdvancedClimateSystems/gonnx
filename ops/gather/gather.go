package gather

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var gatherTypeConstraints = [][]tensor.Dtype{
	ops.AllTypes,
	{tensor.Int32, tensor.Int64},
}

// Gather represents the ONNX gather operator.
type Gather struct {
	ops.BaseOperator

	axis int // axis to gather on, default is 0
}

// newGather creates a new gather operator.
func newGather(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Gather{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"gather",
		),
		axis: 0,
	}
}

// Init initializes the gather operator.
func (g *Gather) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) == 1 {
		attr := attributes[0]

		if attr.GetName() == axis {
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
func (g *Gather) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
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

// Perform gather according to the definition given by ONNX :
// --------------------------
// For  axis = 0 :
// Let  k = indices[i_{0}, ..., i_{q-1}]
// Then output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]
//
// For  axis = 1 :
// Let  k = indices[i_{0}, ..., i_{q-1}]
// Then output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]
// --------------------------
// where q: size of `indices`
//
//	r: size of `data`
//	i and j are here indices which should be iterated over.
//
// A simplified example of how i and j work in such a statement (not related to gather):
// suppose x = [1, 2] and y = [4, 5], and we have statement:
//
//	l = x[i_0]
//	output[i_0, j_0] = y[j_0] - l
//
// This means, for each valid combination of (i_0, j_0) (in this case (0,0) (0,1), (1,0) (1,1) )
// we evaluate the expression, so:
//
//	l = x[0]                   -> l = 1
//	output[0, 0] = y[0] - l    -> output[0,0] = 4 - 1 = 3
//	l = x[0]                   -> l = 1
//	output[0, 1] = y[1] - l    -> output[0,1] = 5 - 1 = 4
//	l = x[1]                   -> l = 2
//	output[1, 0] = y[0] - l    -> output[1,0] = 4 - 2 = 2
//	l = x[1]                   -> l = 2
//	output[1, 1] = y[1] - l    -> output[1,1] = 5 - 2 = 3
//
// so this results in:
//
//	output = [ 3  4 ]
//	         [ 2  3 ]
//
// -------------------------
// The implementation iterates over each element in 'indices', and k is extracted.
// For each given k (and therefore also [i_0, ..., i_q-1]) we need to iterate over each combination
// of [j_0, ..., j_r-1] and perform the above assignment. Instead of explicitly iterating, we use
// slicing to extract the blocks that we need to assign, and then pairwise assign them.
func gather(out, data, indices tensor.Tensor, axis int) error {
	it := indices.Iterator()
	it.Reset()

	for !it.Done() {
		coords := it.Coord()

		at, err := indices.At(coords...)
		if err != nil {
			return err
		}

		k, ok := at.(int)
		if !ok {
			return ops.ErrTypeAssert("int", at)
		}

		// Slice that selects `k` on the given axis.
		// Equivalent to: data[:, ... , :, k, :, ..., :], where `k` is on the index `axis`
		dslices := make([]tensor.Slice, len(data.Shape()))
		dslices[axis] = ops.NewSlicer(k)
		dataSlice, _ := data.Slice(dslices...)

		// slice with the current indices (used to make k) starting from `axis` and
		// the rest nil.
		// Equivalent to:
		//    out[:, ... , :, i_1, ..., i_N, :, ..., :]
		// where i_1 starts at index 'axis'. Note that:  k = indices[i_1, ..., i_N]
		oslices := make([]tensor.Slice, len(coords)+len(data.Shape())-1)
		for i, s := range coords {
			oslices[i+axis] = ops.NewSlicer(s)
		}

		outputSlice, _ := out.Slice(oslices...)

		err = ops.PairwiseAssign(outputSlice, dataSlice)
		if err != nil {
			return err
		}

		_, err = it.Next()
		if err != nil {
			return err
		}
	}

	return nil
}

// insertWithReplace makes a new array, which is equal to an insertion of all elements of `a`
// into `x` at index `axis`. The element at x[axis] is removed (i.e. it is replaced with `a`).
// Output array always has length: len(a) + len(x) - 1
// Example:
// > a = [-1, -2, -3]
// > x = [1, 2, 3, 4, 5, 6, 7]
// insertWithReplace(a, x, 3) -> [1, 2, 3, -1, -2, -3, 5, 6, 7].
func insertWithReplace(a, x []int, axis int) []int {
	y := append([]int{}, x[:axis]...)
	y = append(y, a...)

	if axis+1 < len(x) {
		y = append(y, x[axis+1:]...)
	}

	return y
}
