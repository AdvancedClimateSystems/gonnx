package tensor

// DataType identifies the element type of a tensor. The values are those of the ONNX
// TensorProto.DataType enum, so converting between the two is a plain conversion.
type DataType int32

// The data types a tensor can hold.
const (
	Undefined DataType = 0
	Float32   DataType = 1
	Uint8     DataType = 2
	Int8      DataType = 3
	Uint16    DataType = 4
	Int16     DataType = 5
	Int32     DataType = 6
	Int64     DataType = 7
	Bool      DataType = 9
	Float64   DataType = 11
	Uint32    DataType = 12
	Uint64    DataType = 13
)

var dataTypeNames = map[DataType]string{
	Float32: "float32",
	Uint8:   "uint8",
	Int8:    "int8",
	Uint16:  "uint16",
	Int16:   "int16",
	Int32:   "int32",
	Int64:   "int64",
	Bool:    "bool",
	Float64: "float64",
	Uint32:  "uint32",
	Uint64:  "uint64",
}

// String returns the name of the Go type that holds elements of this data type.
func (d DataType) String() string {
	if name, ok := dataTypeNames[d]; ok {
		return name
	}

	return "undefined"
}

// IsSupported reports whether a tensor can hold elements of this data type.
func (d DataType) IsSupported() bool {
	_, ok := dataTypeNames[d]

	return ok
}

// Float is the set of floating point element types.
type Float interface {
	float32 | float64
}

// Integer is the set of integer element types.
type Integer interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// Numeric is the set of element types that support arithmetic.
type Numeric interface {
	Integer | Float
}

// Element is the set of all element types a tensor can hold.
type Element interface {
	Numeric | bool
}

// DataTypeOf returns the data type that corresponds to the element type T.
func DataTypeOf[T Element]() DataType {
	return dataTypeOf[T]()
}

func dataTypeOf[T Element]() DataType {
	var zero T

	switch any(zero).(type) {
	case float32:
		return Float32
	case float64:
		return Float64
	case int8:
		return Int8
	case int16:
		return Int16
	case int32:
		return Int32
	case int64:
		return Int64
	case uint8:
		return Uint8
	case uint16:
		return Uint16
	case uint32:
		return Uint32
	case uint64:
		return Uint64
	default:
		return Bool
	}
}

// Shape is the size of each dimension of a tensor.
type Shape []int

// Size returns the number of elements in a tensor of this shape. An empty shape is a
// scalar with one element.
func (s Shape) Size() int {
	size := 1
	for _, dim := range s {
		size *= dim
	}

	return size
}

// Equal reports whether both shapes have the same dimensions.
func (s Shape) Equal(other Shape) bool {
	if len(s) != len(other) {
		return false
	}

	for i := range s {
		if s[i] != other[i] {
			return false
		}
	}

	return true
}

// Clone returns a copy of the shape.
func (s Shape) Clone() Shape {
	return append(make(Shape, 0, len(s)), s...)
}
