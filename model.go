package gonnx

import (
	"archive/zip"
	"io"
	"os"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"google.golang.org/protobuf/proto"
	"gorgonia.org/tensor"
)

// Tensors is a map with tensors.
type Tensors map[string]tensor.Tensor

// Model defines a model that can be used for inference.
type Model struct {
	mp         *onnx.ModelProto
	parameters Tensors
	Opset      Opset
}

// NewModelFromFile creates a new model from a path to a file.
func NewModelFromFile(path string) (*Model, error) {
	bytesModel, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	return NewModelFromBytes(bytesModel)
}

// NewModelFromZipFile creates a new model from a file in a zip archive.
func NewModelFromZipFile(file *zip.File) (*Model, error) {
	fc, err := file.Open()
	if err != nil {
		return nil, err
	}

	bytesModel, err := io.ReadAll(fc)
	if err != nil {
		return nil, err
	}

	return NewModelFromBytes(bytesModel)
}

// NewModelFromBytes creates a new model from a list of bytes.
func NewModelFromBytes(bytesModel []byte) (*Model, error) {
	mp, err := ModelProtoFromBytes(bytesModel)
	if err != nil {
		return nil, err
	}

	return NewModel(mp)
}

// NewModel creates a new model ready for inference given a path to an onnx file.
func NewModel(mp *onnx.ModelProto) (*Model, error) {
	params, err := mp.Graph.Params()
	if err != nil {
		return nil, err
	}

	opsetImports := mp.GetOpsetImport()

	var opsetID int64

	for i := 0; i < len(opsetImports); i++ {
		version := opsetImports[i].GetVersion()
		if version > opsetID {
			opsetID = version
		}
	}

	opset, err := ResolveOpset(opsetID)
	if err != nil {
		return nil, err
	}

	return &Model{
		mp:         mp,
		parameters: params,
		Opset:      opset,
	}, nil
}

// ModelProtoFromBytes creates an onnx.ModelProto based on a list of bytes.
func ModelProtoFromBytes(bytesModel []byte) (*onnx.ModelProto, error) {
	mp := &onnx.ModelProto{}
	if err := proto.Unmarshal(bytesModel, mp); err != nil {
		return nil, err
	}

	return mp, nil
}

// InputNames returns this models input names as defined by the model proto.
func (m *Model) InputNames() []string {
	return m.mp.Graph.InputNames()
}

// InputShapes returns the shapes for all input tensors.
func (m *Model) InputShapes() onnx.Shapes {
	return m.mp.Graph.InputShapes()
}

// InputDimSize returns the size of the input dimension given an input tensor.
func (m *Model) InputDimSize(input string, i int) (int, error) {
	if !m.hasInput(input) {
		return 0, ErrModel("input %v does not exist", input)
	}

	inputShape := m.mp.Graph.InputShapes()[input]

	if i >= len(inputShape) {
		return 0, ErrModel("input %v only has %d dimensions, but index %d was required", input, len(inputShape), i)
	}

	return int(inputShape[i].Size), nil
}

// OutputNames returns this models output names as defined by the model proto.
func (m *Model) OutputNames() []string {
	return m.mp.Graph.OutputNames()
}

// OutputShapes returns the shapes for all output tensors.
func (m *Model) OutputShapes() onnx.Shapes {
	return m.mp.Graph.OutputShapes()
}

// OutputShape returns the shape of a specific output tensors.
func (m *Model) OutputShape(output string) onnx.Shape {
	return m.mp.Graph.OutputShapes()[output]
}

// ParamNames returns this models parameter names as defined by the model proto.
func (m *Model) ParamNames() []string {
	return m.mp.Graph.ParamNames()
}

func (m *Model) hasInput(input string) bool {
	for _, inputName := range m.InputNames() {
		if inputName == input {
			return true
		}
	}

	return false
}

// Run builds and executes the computional graph of the network given the inputs.
func (m *Model) Run(inputs Tensors) (Tensors, error) {
	if err := m.validateShapes(inputs); err != nil {
		return nil, err
	}

	tensors := make(Tensors)
	for inputName, inputTensor := range inputs {
		tensors[inputName] = inputTensor
	}

	for parameterName, parameterTensor := range m.parameters {
		tensors[parameterName] = parameterTensor
	}

	for _, n := range m.mp.Graph.GetNode() {
		op, ok := m.Opset[n.GetOpType()]
		if !ok {
			return nil, ops.ErrUnknownOperatorType(n.GetOpType())
		}

		if err := m.applyOp(op(), n, tensors); err != nil {
			return nil, err
		}
	}

	outputTensors := make(Tensors)
	for _, outputName := range m.OutputNames() {
		outputTensors[outputName] = tensors[outputName]
	}

	return outputTensors, nil
}

// applyOp applies the operation to the graph.
func (m *Model) applyOp(op ops.Operator, n *onnx.NodeProto, tensors Tensors) error {
	if err := op.Init(n); err != nil {
		return err
	}

	inputTensors, err := getInputTensorsForNode(n.GetInput(), tensors)
	if err != nil {
		return err
	}

	inputTensors, err = op.ValidateInputs(inputTensors)
	if err != nil {
		return err
	}

	outputTensors, err := op.Apply(inputTensors)
	if err != nil {
		return err
	}

	return setOutputTensorsOfNode(n.GetOutput(), outputTensors, tensors)
}

// validateShapes validates if the tensors passed in have the same shape as the shapes defined
// by the onnx.Shapes.
func (m *Model) validateShapes(inputTensors Tensors) error {
	for name, shapeExpected := range m.InputShapes() {
		// If the input is a parameter, the user does not have to provide a tensor for it.
		if _, ok := m.parameters[name]; ok {
			continue
		}

		tensor, ok := inputTensors[name]
		if !ok {
			return ErrModel("tensor: %v not found", name)
		}

		shapeReceived := tensor.Shape()

		if len(shapeReceived) != len(shapeExpected) {
			return ErrInvalidShape(shapeExpected, shapeReceived)
		}

		for i, dim := range shapeExpected {
			// because the dimension is dynamic, it can have any size
			// and we do not have to check for it
			if dim.IsDynamic {
				continue
			}

			if dim.Size != int64(shapeReceived[i]) {
				return ErrInvalidShape(shapeExpected, shapeReceived)
			}
		}
	}

	return nil
}

func getInputTensorsForNode(names []string, tensors Tensors) ([]tensor.Tensor, error) {
	var inputTensors []tensor.Tensor

	for _, tensorName := range names {
		// An empty name can happen in between optional inputs, like:
		//   [<required_input>, <optional_input>, nil, <optional_input>]
		// In such a case, ONNX includes the name of the input in the node, and we need
		// to set a value (nil) for it, although it will not be used.
		if tensorName == "" {
			inputTensors = append(inputTensors, nil)
		} else if tensor, ok := tensors[tensorName]; ok {
			inputTensors = append(inputTensors, tensor)
		} else {
			return nil, ErrModel("no tensor yet for name %v", tensorName)
		}
	}

	return inputTensors, nil
}

func setOutputTensorsOfNode(
	names []string, outputTensors []tensor.Tensor, tensors Tensors,
) error {
	if len(names) != len(outputTensors) {
		return ErrModel("could not set output tensor")
	}

	for i, tensor := range outputTensors {
		tensors[names[i]] = tensor
	}

	return nil
}
