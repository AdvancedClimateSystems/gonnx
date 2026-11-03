package ops

import (
	"github.com/advancedclimatesystems/gonnx/tensor"
)

// DTypeToONNXType returns the ONNX TensorProto data type of a tensor data type.
func DTypeToONNXType(t tensor.DataType) (int32, error) {
	if !t.IsSupported() {
		return 0, ErrUnknownTensorONNXDtype(t)
	}

	return int32(t), nil
}

// ConvertTensorDtype converts a tensor to a tensor of the ONNX data type newType.
func ConvertTensorDtype(t *tensor.Tensor, newType int32) (*tensor.Tensor, error) {
	shape := t.Shape()

	switch data := t.Data().(type) {
	case []float32:
		return convertTensor(data, shape, newType)
	case []float64:
		return convertTensor(data, shape, newType)
	case []int8:
		return convertTensor(data, shape, newType)
	case []int16:
		return convertTensor(data, shape, newType)
	case []int32:
		return convertTensor(data, shape, newType)
	case []int64:
		return convertTensor(data, shape, newType)
	case []uint8:
		return convertTensor(data, shape, newType)
	case []uint16:
		return convertTensor(data, shape, newType)
	case []uint32:
		return convertTensor(data, shape, newType)
	case []uint64:
		return convertTensor(data, shape, newType)
	case []bool:
		return convertBooleanTensor(data, shape, newType)
	default:
		return nil, ErrConversionInvalidType(t.DataType(), newType)
	}
}

func convertTensor[B NumericType](backing []B, shape tensor.Shape, dataType int32) (*tensor.Tensor, error) {
	switch tensor.DataType(dataType) {
	case tensor.Float32:
		return tensor.New(createNewBacking[B, float32](backing), shape...), nil
	case tensor.Float64:
		return tensor.New(createNewBacking[B, float64](backing), shape...), nil
	case tensor.Int8:
		return tensor.New(createNewBacking[B, int8](backing), shape...), nil
	case tensor.Int16:
		return tensor.New(createNewBacking[B, int16](backing), shape...), nil
	case tensor.Int32:
		return tensor.New(createNewBacking[B, int32](backing), shape...), nil
	case tensor.Int64:
		return tensor.New(createNewBacking[B, int64](backing), shape...), nil
	case tensor.Uint8:
		return tensor.New(createNewBacking[B, uint8](backing), shape...), nil
	case tensor.Uint16:
		return tensor.New(createNewBacking[B, uint16](backing), shape...), nil
	case tensor.Uint32:
		return tensor.New(createNewBacking[B, uint32](backing), shape...), nil
	case tensor.Uint64:
		return tensor.New(createNewBacking[B, uint64](backing), shape...), nil
	case tensor.Bool:
		return tensor.New(createNewBooleanBacking(backing), shape...), nil
	default:
		return nil, ErrConversionNotSupported(dataType)
	}
}

func convertBooleanTensor(backing []bool, shape tensor.Shape, dataType int32) (*tensor.Tensor, error) {
	switch tensor.DataType(dataType) {
	case tensor.Float32:
		return tensor.New(createNewBackingFromBoolean[float32](backing), shape...), nil
	case tensor.Float64:
		return tensor.New(createNewBackingFromBoolean[float64](backing), shape...), nil
	case tensor.Int8:
		return tensor.New(createNewBackingFromBoolean[int8](backing), shape...), nil
	case tensor.Int16:
		return tensor.New(createNewBackingFromBoolean[int16](backing), shape...), nil
	case tensor.Int32:
		return tensor.New(createNewBackingFromBoolean[int32](backing), shape...), nil
	case tensor.Int64:
		return tensor.New(createNewBackingFromBoolean[int64](backing), shape...), nil
	case tensor.Uint8:
		return tensor.New(createNewBackingFromBoolean[uint8](backing), shape...), nil
	case tensor.Uint16:
		return tensor.New(createNewBackingFromBoolean[uint16](backing), shape...), nil
	case tensor.Uint32:
		return tensor.New(createNewBackingFromBoolean[uint32](backing), shape...), nil
	case tensor.Uint64:
		return tensor.New(createNewBackingFromBoolean[uint64](backing), shape...), nil
	case tensor.Bool:
		return tensor.New(append([]bool(nil), backing...), shape...), nil
	default:
		return nil, ErrConversionNotSupported(dataType)
	}
}

func createNewBacking[B NumericType, R NumericType](backing []B) []R {
	newBacking := make([]R, len(backing))
	for i := range backing {
		newBacking[i] = R(backing[i])
	}

	return newBacking
}

func createNewBooleanBacking[B NumericType](backing []B) []bool {
	newBacking := make([]bool, len(backing))
	for i := range backing {
		newBacking[i] = backing[i] != 0
	}

	return newBacking
}

func createNewBackingFromBoolean[T NumericType](backing []bool) []T {
	newBacking := make([]T, len(backing))

	for i := range backing {
		if backing[i] {
			newBacking[i] = 1
		}
	}

	return newBacking
}
