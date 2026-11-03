package ops

import (
	"fmt"
	"math/rand"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// InputFixture is a function that generates inputs for ops. Useful in testing.
type InputFixture func() []*tensor.Tensor

// Float32TensorFixture returns a float32 tensor with the given shape, filled with
// 0, 1, 2, ... in row-major order.
func Float32TensorFixture(shp ...int) *tensor.Tensor {
	return tensor.New(Arange(NElements(shp...), 1), shp...)
}

func RandomFloat32TensorFixture(r *rand.Rand, shp ...int) *tensor.Tensor {
	rands := make([]float32, NElements(shp...))
	for i := 0; i < NElements(shp...); i++ {
		rands[i] = r.Float32()
	}

	return tensor.New(rands, shp...)
}

// TensorWithBackingFixture returns a tensor with the given backing, a typed slice such as
// []float32, and shape.
func TensorWithBackingFixture(b any, shp ...int) *tensor.Tensor {
	switch backing := b.(type) {
	case []float32:
		return tensor.New(backing, shp...)
	case []float64:
		return tensor.New(backing, shp...)
	case []int8:
		return tensor.New(backing, shp...)
	case []int16:
		return tensor.New(backing, shp...)
	case []int32:
		return tensor.New(backing, shp...)
	case []int64:
		return tensor.New(backing, shp...)
	case []uint8:
		return tensor.New(backing, shp...)
	case []uint16:
		return tensor.New(backing, shp...)
	case []uint32:
		return tensor.New(backing, shp...)
	case []uint64:
		return tensor.New(backing, shp...)
	case []bool:
		return tensor.New(backing, shp...)
	default:
		panic(fmt.Sprintf("unsupported tensor backing %T", b))
	}
}

// TensorInputsFixture returns a list with a given number of tensors.
func TensorInputsFixture(nTensors int) []*tensor.Tensor {
	result := make([]*tensor.Tensor, nTensors)
	for i := 0; i < nTensors; i++ {
		result[i] = tensor.New([]float32{0.0}, 1)
	}

	return result
}

// EmptyNodeProto returns a node proto with no attributes.
func EmptyNodeProto() *onnx.NodeProto {
	return &onnx.NodeProto{Attribute: []*onnx.AttributeProto{}}
}
