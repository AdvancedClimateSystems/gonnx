package gonnx

import (
	"archive/zip"
	"io"
	"os"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
	"google.golang.org/protobuf/proto"
)

// Tensors is a map with tensors.
type Tensors map[string]*tensor.Tensor

// Model defines a model that can be used for inference.
type Model struct {
	mp *onnx.ModelProto

	// Every tensor name used in the graph is given a slot, so Run can keep the tensors in
	// a slice instead of a map.
	nSlots         int
	nodes          []node
	parameters     []slotTensor
	externalInputs []namedSlot
	outputs        []namedSlot
	inputShapes    []namedShape
	maxNodeInputs  int
}

// node is a node of the graph together with its operator, initialized with the
// attributes of the node.
type node struct {
	operator   ops.Operator
	inputs     []int
	inputNames []string
	outputs    []int
}

// noSlot marks an omitted optional input or output of a node.
const noSlot = -1

type slotTensor struct {
	slot   int
	tensor *tensor.Tensor
}

type namedSlot struct {
	name string
	slot int
}

type namedShape struct {
	name  string
	shape onnx.Shape
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

	resolved, err := resolveOpset(opsetID)
	if err != nil {
		return nil, err
	}

	m := &Model{mp: mp}
	slots := map[string]int{}

	slotOf := func(name string) int {
		if name == "" {
			return noSlot
		}

		slot, ok := slots[name]
		if !ok {
			slot = len(slots)
			slots[name] = slot
		}

		return slot
	}

	// Parameters are assigned after the inputs passed to Run, so a parameter wins when
	// both have the same name.
	for name, t := range params {
		m.parameters = append(m.parameters, slotTensor{slot: slotOf(name), tensor: t})
	}

	produced := map[string]bool{}

	for _, n := range mp.Graph.GetNode() {
		newOperator, ok := resolved[n.GetOpType()]
		if !ok {
			return nil, ops.ErrUnknownOperatorType(n.GetOpType())
		}

		operator := newOperator()
		if err := operator.Init(n); err != nil {
			return nil, err
		}

		if preparer, ok := operator.(ops.Preparer); ok {
			constants := make([]*tensor.Tensor, len(n.GetInput()))
			for i, name := range n.GetInput() {
				constants[i] = params[name]
			}

			preparer.Prepare(constants)
		}

		graphNode := node{operator: operator, inputNames: n.GetInput()}

		for _, name := range n.GetInput() {
			_, isParameter := params[name]
			if name != "" && !isParameter && !produced[name] {
				if _, seen := slots[name]; !seen {
					m.externalInputs = append(m.externalInputs, namedSlot{name: name, slot: slotOf(name)})
				}
			}

			graphNode.inputs = append(graphNode.inputs, slotOf(name))
		}

		for _, name := range n.GetOutput() {
			graphNode.outputs = append(graphNode.outputs, slotOf(name))
			produced[name] = true
		}

		m.maxNodeInputs = max(m.maxNodeInputs, len(n.GetInput()), operator.GetMaxInputs())
		m.nodes = append(m.nodes, graphNode)
	}

	for _, name := range mp.Graph.OutputNames() {
		_, isParameter := params[name]
		if _, seen := slots[name]; !seen && !isParameter {
			m.externalInputs = append(m.externalInputs, namedSlot{name: name, slot: slotOf(name)})
		}

		m.outputs = append(m.outputs, namedSlot{name: name, slot: slotOf(name)})
	}

	inputShapes := mp.Graph.InputShapes()

	for _, name := range mp.Graph.InputNames() {
		if _, isParameter := params[name]; !isParameter {
			m.inputShapes = append(m.inputShapes, namedShape{name: name, shape: inputShapes[name]})
		}
	}

	m.nSlots = len(slots)

	return m, nil
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

	values := make([]*tensor.Tensor, m.nSlots)

	for _, input := range m.externalInputs {
		values[input.slot] = inputs[input.name]
	}

	for _, parameter := range m.parameters {
		values[parameter.slot] = parameter.tensor
	}

	nodeInputs := make([]*tensor.Tensor, 0, m.maxNodeInputs)

	for _, n := range m.nodes {
		if err := applyOp(n, values, nodeInputs[:0]); err != nil {
			return nil, err
		}
	}

	outputTensors := make(Tensors, len(m.outputs))
	for _, output := range m.outputs {
		outputTensors[output.name] = values[output.slot]
	}

	return outputTensors, nil
}

// applyOp applies the operator of a node to its input tensors, and stores the results in
// the slots of its outputs. The input tensors are collected in nodeInputs.
func applyOp(n node, values, nodeInputs []*tensor.Tensor) error {
	for i, slot := range n.inputs {
		if slot == noSlot {
			nodeInputs = append(nodeInputs, nil)

			continue
		}

		if values[slot] == nil {
			return ErrModel("no tensor yet for name %v", n.inputNames[i])
		}

		nodeInputs = append(nodeInputs, values[slot])
	}

	nodeInputs, err := n.operator.ValidateInputs(nodeInputs)
	if err != nil {
		return err
	}

	outputTensors, err := n.operator.Apply(nodeInputs)
	if err != nil {
		return err
	}

	if len(n.outputs) != len(outputTensors) {
		return ErrModel("could not set output tensor")
	}

	for i, slot := range n.outputs {
		if slot != noSlot {
			values[slot] = outputTensors[i]
		}
	}

	return nil
}

// validateShapes validates if the tensors passed in have the same shape as the shapes
// of the graph inputs that are not parameters.
func (m *Model) validateShapes(inputTensors Tensors) error {
	for _, input := range m.inputShapes {
		tensor, ok := inputTensors[input.name]
		if !ok {
			return ErrModel("tensor: %v not found", input.name)
		}

		shapeExpected := input.shape
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
