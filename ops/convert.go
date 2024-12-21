package ops

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"gorgonia.org/tensor"
)

// Number is a type which represents a number.
type Number interface {
	float32 | float64 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// ConvertTensorDtype converts an interface of a specific dtype to a new dtype.
func ConvertTensorDtype(t tensor.Tensor, newType int32) (tensor.Tensor, error) {
	var (
		err        error
		newBacking any
	)

	backing := IfScalarToSlice(t.Data())

	switch t.Dtype() {
	case tensor.Float32:
		newBacking, err = convertBacking(backing.([]float32), newType)
	case tensor.Float64:
		newBacking, err = convertBacking(backing.([]float64), newType)
	case tensor.Int8:
		newBacking, err = convertBacking(backing.([]int8), newType)
	case tensor.Int16:
		newBacking, err = convertBacking(backing.([]int16), newType)
	case tensor.Int32:
		newBacking, err = convertBacking(backing.([]int32), newType)
	case tensor.Int64:
		newBacking, err = convertBacking(backing.([]int64), newType)
	case tensor.Uint8:
		newBacking, err = convertBacking(backing.([]uint8), newType)
	case tensor.Uint16:
		newBacking, err = convertBacking(backing.([]uint16), newType)
	case tensor.Uint32:
		newBacking, err = convertBacking(backing.([]uint32), newType)
	case tensor.Uint64:
		newBacking, err = convertBacking(backing.([]uint64), newType)
	default:
		return nil, ErrConversionInvalidType(t.Dtype(), newType)
	}

	if err != nil {
		return nil, err
	}

	return tensor.New(tensor.WithShape(t.Shape()...), tensor.WithBacking(newBacking)), nil
}

func DTypeToONNXType(t tensor.Dtype) (int32, error) {
	switch t {
	case tensor.Float32:
		return int32(onnx.TensorProto_FLOAT), nil
	case tensor.Float64:
		return int32(onnx.TensorProto_DOUBLE), nil
	case tensor.Int8:
		return int32(onnx.TensorProto_INT8), nil
	case tensor.Int16:
		return int32(onnx.TensorProto_INT16), nil
	case tensor.Int32:
		return int32(onnx.TensorProto_INT32), nil
	case tensor.Int64:
		return int32(onnx.TensorProto_INT64), nil
	case tensor.Uint8:
		return int32(onnx.TensorProto_UINT8), nil
	case tensor.Uint16:
		return int32(onnx.TensorProto_UINT16), nil
	case tensor.Uint32:
		return int32(onnx.TensorProto_UINT32), nil
	case tensor.Uint64:
		return int32(onnx.TensorProto_UINT64), nil
	default:
		return 0, ErrUnknownTensorONNXDtype(t)
	}
}

func convertBacking[B Number](backing []B, dataType int32) (any, error) {
	switch onnx.TensorProto_DataType(dataType) {
	case onnx.TensorProto_FLOAT:
		return createNewBacking[B, float32](backing), nil
	case onnx.TensorProto_DOUBLE:
		return createNewBacking[B, float64](backing), nil
	case onnx.TensorProto_INT8:
		return createNewBacking[B, int8](backing), nil
	case onnx.TensorProto_INT16:
		return createNewBacking[B, int16](backing), nil
	case onnx.TensorProto_INT32:
		return createNewBacking[B, int32](backing), nil
	case onnx.TensorProto_INT64:
		return createNewBacking[B, int64](backing), nil
	case onnx.TensorProto_UINT8:
		return createNewBacking[B, uint8](backing), nil
	case onnx.TensorProto_UINT16:
		return createNewBacking[B, uint16](backing), nil
	case onnx.TensorProto_UINT32:
		return createNewBacking[B, uint32](backing), nil
	case onnx.TensorProto_UINT64:
		return createNewBacking[B, uint64](backing), nil
	case onnx.TensorProto_BFLOAT16, onnx.TensorProto_BOOL, onnx.TensorProto_COMPLEX64, onnx.TensorProto_COMPLEX128, onnx.TensorProto_FLOAT16, onnx.TensorProto_UNDEFINED, onnx.TensorProto_STRING:
		return nil, ErrConversionNotSupported(dataType)
	default:
		return nil, ErrConversionNotSupported(dataType)
	}
}

func createNewBacking[B Number, R Number](backing []B) []R {
	newBacking := make([]R, len(backing))
	for i := range backing {
		newBacking[i] = R(backing[i])
	}

	return newBacking
}
