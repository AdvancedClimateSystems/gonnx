// Package onnx provides wrapper functionality around the onnx.proto3.pb.go
// The goal is to provide a stable(ish) api around the proto file
// since the proto file is generated based on onnx.proto.
package onnx

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"gorgonia.org/tensor"
)

// InputNames returns the input names for a GraphProto.
func (g *GraphProto) InputNames() []string {
	return getNamesFromValueProto(g.GetInput())
}

// InputShapes returns the input shapes for a GraphProto.
func (g *GraphProto) InputShapes() Shapes {
	return getShapesFromValueProto(g.GetInput())
}

// OutputNames returns the output names for a GraphProto.
func (g *GraphProto) OutputNames() []string {
	return getNamesFromValueProto(g.GetOutput())
}

// OutputShapes returns the output shapes for a GraphProto.
func (g *GraphProto) OutputShapes() Shapes {
	return getShapesFromValueProto(g.GetOutput())
}

// ParamNames returns the names of the parameters as defined by the GraphProto.
func (g *GraphProto) ParamNames() []string {
	return getNamesFromTensorProto(g.GetInitializer())
}

// Params returns the parameters of the graph proto as a map of tensors.
func (g *GraphProto) Params() (map[string]tensor.Tensor, error) {
	initializers := g.GetInitializer()
	res := make(map[string]tensor.Tensor, len(initializers))

	for _, i := range initializers {
		t, err := TensorFromProto(i)
		if err != nil {
			return nil, err
		}

		res[i.Name] = t
	}

	return res, nil
}

// Shapes contains the shapes for different named tensors.
type Shapes map[string]Shape

// Shape is a list of dimensions.
type Shape []Dim

// String prints a shape in a human-friendly matter.
func (s Shape) String() string {
	dimSizes := make([]int64, 0, len(s))

	for _, dim := range s {
		dimSizes = append(dimSizes, dim.Size)
	}

	return fmt.Sprintf("%d", dimSizes)
}

var ErrInvalidType = errors.New("invalid type")

// Dim is a dimension.
type Dim struct {
	IsDynamic bool
	Name      string
	Size      int64
}

// String return's a string representation of the dimension.
func (s Dim) String() string {
	return fmt.Sprintf("dynamic: %v, name: %v, size: %v\n", s.IsDynamic, s.Name, s.Size)
}

func getNamesFromValueProto(protos []*ValueInfoProto) []string {
	res := make([]string, len(protos))

	for i, output := range protos {
		res[i] = output.GetName()
	}

	return res
}

func getShapesFromValueProto(protos []*ValueInfoProto) Shapes {
	if protos == nil {
		return map[string]Shape{}
	}

	shapes := make(map[string]Shape, len(protos))

	for _, p := range protos {
		t := p.GetType()
		if t == nil {
			continue
		}

		tt := t.GetTensorType()
		if tt == nil {
			continue
		}

		shapeProto := tt.GetShape()
		if shapeProto == nil {
			continue
		}

		dims := shapeProto.GetDim()
		if dims == nil {
			continue
		}

		shape := make([]Dim, len(dims))

		for i, dim := range dims {
			param := dim.GetDimParam()
			v := dim.GetDimValue()

			isDynamic := false
			if v == 0 {
				isDynamic = true
			}

			shape[i] = Dim{IsDynamic: isDynamic, Name: param, Size: v}
		}

		shapes[p.GetName()] = shape
	}

	return shapes
}

func getNamesFromTensorProto(protos []*TensorProto) []string {
	res := make([]string, len(protos))

	for i, output := range protos {
		res[i] = output.GetName()
	}

	return res
}

// TensorFromProto returns a tensor.Tensor from an onnx.TensorProto.
func TensorFromProto(tp *TensorProto) (tensor.Tensor, error) {
	var (
		values interface{}
		err    error
	)

	typeMap := TensorProto_DataType_value

	switch tp.DataType {
	case typeMap["FLOAT"]:
		values, err = getFloatData(tp)
	case typeMap["UINT8"]:
		values, err = getUint8Data(tp)
	case typeMap["INT8"]:
		values, err = getInt8Data(tp)
	case typeMap["UINT16"]:
		values, err = getUint16Data(tp)
	case typeMap["INT16"]:
		values, err = getInt16Data(tp)
	case typeMap["UINT32"]:
		values, err = getUint32Data(tp)
	case typeMap["INT32"]:
		values, err = getInt32Data(tp)
	case typeMap["UINT64"]:
		values, err = getUint64Data(tp)
	case typeMap["INT64"]:
		values, err = getInt64Data(tp)
	case typeMap["DOUBLE"]:
		values, err = getDoubleData(tp)
	case typeMap["BOOL"]:
		values = getBoolData(tp)
	default:
		// At this moment the datatype is either UNDEFINED or some datatype we currently
		// do not support.
		switch {
		case len(tp.FloatData) > 0:
			values, err = getFloatData(tp)
		case len(tp.Int32Data) > 0:
			values, err = getInt32Data(tp)
		case len(tp.Int64Data) > 0:
			values, err = getInt64Data(tp)
		case len(tp.DoubleData) > 0:
			values, err = getDoubleData(tp)
		case len(tp.Uint64Data) > 0:
			values, err = getUint64Data(tp)
		default:
			return nil, ErrInvalidType
		}
	}

	if err != nil {
		return nil, err
	}

	return tensor.New(tensor.WithShape(getDims(tp)...), tensor.WithBacking(values)), nil
}

func getFloatData(tp *TensorProto) ([]float32, error) {
	if len(tp.FloatData) > 0 {
		return tp.GetFloatData(), nil
	}

	return ReadFloat32ArrayFromBytes(tp.RawData)
}

func getInt8Data(tp *TensorProto) ([]int8, error) {
	if len(tp.Int32Data) > 0 {
		return Int32ArrayToInt8Array(tp.GetInt32Data()), nil
	}

	return ReadInt8ArrayFromBytes(tp.RawData)
}

func getUint8Data(tp *TensorProto) ([]uint8, error) {
	if len(tp.Int32Data) > 0 {
		return Int32ArrayToUint8Array(tp.GetInt32Data()), nil
	}

	return ReadUint8ArrayFromBytes(tp.RawData)
}

func getUint16Data(tp *TensorProto) ([]uint16, error) {
	if len(tp.Int32Data) > 0 {
		return Int32ArrayToUint16Array(tp.GetInt32Data()), nil
	}

	return ReadUint16ArrayFromBytes(tp.RawData)
}

func getInt16Data(tp *TensorProto) ([]int16, error) {
	if len(tp.Int32Data) > 0 {
		return Int32ArrayToInt16Array(tp.GetInt32Data()), nil
	}

	return ReadInt16ArrayFromBytes(tp.RawData)
}

func getUint32Data(tp *TensorProto) ([]uint32, error) {
	if len(tp.Uint64Data) > 0 {
		return Uint64ArrayToUint32Array(tp.GetUint64Data()), nil
	}

	return ReadUint32ArrayFromBytes(tp.RawData)
}

func getInt32Data(tp *TensorProto) ([]int32, error) {
	if len(tp.Int32Data) > 0 {
		return tp.GetInt32Data(), nil
	}

	return ReadInt32ArrayFromBytes(tp.RawData)
}

func getUint64Data(tp *TensorProto) ([]uint64, error) {
	if len(tp.Uint64Data) > 0 {
		return tp.GetUint64Data(), nil
	}

	return ReadUint64ArrayFromBytes(tp.RawData)
}

func getInt64Data(tp *TensorProto) ([]int64, error) {
	if len(tp.Int64Data) > 0 {
		return tp.GetInt64Data(), nil
	}

	return ReadInt64ArrayFromBytes(tp.RawData)
}

func getDoubleData(tp *TensorProto) ([]float64, error) {
	if len(tp.DoubleData) > 0 {
		return tp.GetDoubleData(), nil
	}

	return ReadFloat64ArrayFromBytes(tp.RawData)
}

func getBoolData(tp *TensorProto) []bool {
	if len(tp.Int32Data) > 0 {
		return Int32ArrayToBoolArray(tp.GetInt32Data())
	}

	return ReadBoolArrayFromBytes(tp.RawData)
}

const (
	float32Size int = 4
	boolSize    int = 1
	uint8Size   int = 1
	int8Size    int = 1
	uint16Size  int = 2
	int16Size   int = 2
	uint32Size  int = 4
	int32Size   int = 4
	uint64Size  int = 8
	int64Size   int = 8
	float64Size int = 8
)

// ReadFloat32ArrayFromBytes reads data and parses it to an array of float32.
func ReadFloat32ArrayFromBytes(data []byte) ([]float32, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, float32Size)

	var (
		err    error
		values []float32
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != float32Size || err != nil {
			break
		}

		uintElement := binary.LittleEndian.Uint32(element)
		values = append(values, math.Float32frombits(uintElement))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadFloat64ArrayFromBytes reads data and parses it to an array of float64.
func ReadFloat64ArrayFromBytes(data []byte) ([]float64, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, float64Size)

	var (
		err    error
		values []float64
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != float64Size || err != nil {
			break
		}

		uintElement := binary.LittleEndian.Uint64(element)
		values = append(values, math.Float64frombits(uintElement))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadBoolArrayFromBytes reads data and parses it to an array of bool.
// The data is parsed to a bool by comparing the value to 0. If it is
// greater than 0, the bool is considered to be true.
func ReadBoolArrayFromBytes(data []byte) []bool {
	values := make([]bool, len(data))
	for i, b := range data {
		values[i] = b > 0
	}

	return values
}

// ReadUint8ArrayFromBytes reads data and parses it to an array of uint8.
func ReadUint8ArrayFromBytes(data []byte) ([]uint8, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, uint8Size)

	var (
		err    error
		values []uint8
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != uint8Size || err != nil {
			break
		}

		values = append(values, element[0])
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadInt8ArrayFromBytes reads data and parses it to an array of int8.
func ReadInt8ArrayFromBytes(data []byte) ([]int8, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, int8Size)

	var (
		err    error
		values []int8
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != int8Size || err != nil {
			break
		}

		values = append(values, int8(element[0]))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadUint16ArrayFromBytes reads data and parses it to an array of uint16.
func ReadUint16ArrayFromBytes(data []byte) ([]uint16, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, uint16Size)

	var (
		err    error
		values []uint16
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != uint16Size || err != nil {
			break
		}

		values = append(values, binary.LittleEndian.Uint16(element))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadInt16ArrayFromBytes reads data and parses it to an array of int16.
func ReadInt16ArrayFromBytes(data []byte) ([]int16, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, uint16Size)

	var (
		err    error
		values []int16
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != int16Size || err != nil {
			break
		}

		values = append(values, int16(binary.LittleEndian.Uint16(element)))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadUint32ArrayFromBytes reads data and parses it to an array of uint32.
func ReadUint32ArrayFromBytes(data []byte) ([]uint32, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, int32Size)

	var (
		err    error
		values []uint32
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != uint32Size || err != nil {
			break
		}

		values = append(values, binary.LittleEndian.Uint32(element))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadInt32ArrayFromBytes reads data and parses it to an array of int32.
func ReadInt32ArrayFromBytes(data []byte) ([]int32, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, int32Size)

	var (
		err    error
		values []int32
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != int32Size || err != nil {
			break
		}

		values = append(values, int32(binary.LittleEndian.Uint32(element)))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadUint64ArrayFromBytes reads data and parses it to an array of uint64.
func ReadUint64ArrayFromBytes(data []byte) ([]uint64, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, int64Size)

	var (
		err    error
		values []uint64
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != uint64Size || err != nil {
			break
		}

		values = append(values, uint64(binary.LittleEndian.Uint32(element)))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// ReadInt64ArrayFromBytes reads data and parses it to an array of float32.
func ReadInt64ArrayFromBytes(data []byte) ([]int64, error) {
	buffer := bytes.NewReader(data)
	element := make([]byte, int64Size)

	var (
		err    error
		values []int64
	)

	for {
		var n int

		n, err = buffer.Read(element)
		if n != int64Size || err != nil {
			break
		}

		values = append(values, int64(binary.LittleEndian.Uint64(element)))
	}

	if err != io.EOF {
		return nil, err
	}

	return values, nil
}

// Int32ArrayToBoolArray converts an int32 array to a bool array.
// When the value is equal to 1 the boolean is considered to be true.
func Int32ArrayToBoolArray(arr []int32) []bool {
	newArr := make([]bool, len(arr))
	for i, value := range arr {
		newArr[i] = value == 1
	}

	return newArr
}

// Int32ArrayToInt8Array converts an int32 array to an int8 array.
func Int32ArrayToInt8Array(arr []int32) []int8 {
	newArr := make([]int8, len(arr))
	for i, value := range arr {
		newArr[i] = int8(value)
	}

	return newArr
}

// Int32ArrayToUint8Array converts an int32 array to a uint8 array.
func Int32ArrayToUint8Array(arr []int32) []uint8 {
	newArr := make([]uint8, len(arr))
	for i, value := range arr {
		newArr[i] = uint8(value)
	}

	return newArr
}

// Int32ArrayToInt16Array converts an int32 array to a int16 array.
func Int32ArrayToInt16Array(arr []int32) []int16 {
	newArr := make([]int16, len(arr))
	for i, value := range arr {
		newArr[i] = int16(value)
	}

	return newArr
}

// Int32ArrayToUint16Array converts an int32 array to a uint16 array.
func Int32ArrayToUint16Array(arr []int32) []uint16 {
	newArr := make([]uint16, len(arr))
	for i, value := range arr {
		newArr[i] = uint16(value)
	}

	return newArr
}

// Uint64ArrayToUint32Array converts an uint64 array to a uint32 array.
func Uint64ArrayToUint32Array(arr []uint64) []uint32 {
	newArr := make([]uint32, len(arr))
	for i, value := range arr {
		newArr[i] = uint32(value)
	}

	return newArr
}

func getDims(tensor *TensorProto) []int {
	nDims := len(tensor.GetDims())
	dims := make([]int, nDims)

	for i, dimValue := range tensor.GetDims() {
		dims[i] = int(dimValue)
	}

	return dims
}
