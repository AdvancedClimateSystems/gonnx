package lstm

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

const (
	MinLSTMInputs = 3
	MaxLSTMInputs = 8
)

var lstmTypeConstraints = [][]tensor.DataType{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Int32},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

// LSTM represents the ONNX lstm operator.
type LSTM struct {
	ops.BaseOperator

	activationAlpha []float32
	activationBeta  []float32
	activations     []string
	direction       ops.SequenceProcessDirection
	hiddenSize      int
	inputForget     bool

	outputs []string

	inputWeights     *ops.PreparedWeights
	recurrentWeights *ops.PreparedWeights
}

// newLSTM creates a new lstm operator.
func newLSTM(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &LSTM{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinLSTMInputs,
			MaxLSTMInputs,
			typeConstraints,
			"lstm",
		),
		activations: []string{"sigmoid", "tanh", "tanh"},
		direction:   ops.Forward,
		inputForget: false,
		outputs:     []string{"Y", "Y_h", "Y_c"},
	}
}

// Init initializes the lstm operator.
func (l *LSTM) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case ops.ActivationAlphaAttr:
			l.activationAlpha = attr.GetFloats()
		case ops.ActivationBetaAttr:
			l.activationBeta = attr.GetFloats()
		case ops.ActivationsAttr:
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			l.activations = activations
		case ops.ClipAttr:
			return ops.ErrUnsupportedAttribute(attr.GetName(), l)
		case ops.DirectionAttr:
			l.direction = ops.SequenceProcessDirection(attr.GetS())
			if l.direction != ops.Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), l)
			}
		case ops.HiddenSizeAttr:
			l.hiddenSize = int(attr.GetI())
		case "input_forget":
			l.inputForget = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), l)
		}
	}

	l.outputs = n.GetOutput()

	return nil
}

// Apply applies the lstm operator.
func (l *LSTM) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence_lens", l.BaseOperator)
	}

	X, W, R, B, initialH, initialC, P := inputs[0], inputs[1], inputs[2], inputs[3], inputs[5], inputs[6], inputs[7]

	dims, err := ops.ValidateRecurrentInputs(X, W, R, B, initialH, nGates, l.hiddenSize)
	if err != nil {
		return nil, err
	}

	if initialC != nil && !initialC.Shape().Equal(tensor.Shape{1, dims.BatchSize, l.hiddenSize}) {
		return nil, ops.ErrInvalidInput("initial_c must have shape [1, batch_size, hidden_size]", l.BaseOperator)
	}

	if P != nil && !P.Shape().Equal(tensor.Shape{1, nPeepholes * l.hiddenSize}) {
		return nil, ops.ErrInvalidInput("P must have shape [1, 3*hidden_size]", l.BaseOperator)
	}

	var outputs []*tensor.Tensor

	switch X.DataType() {
	case tensor.Float32:
		outputs, err = lstm[float32](l, dims, X, W, R, B, initialH, initialC, P)
	case tensor.Float64:
		outputs, err = lstm[float64](l, dims, X, W, R, B, initialH, initialC, P)
	default:
		return nil, ops.ErrInvalidInputType(0, X.DataType().String(), l.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	// Outputs are positional; a node lists only the outputs it uses.
	return outputs[:min(len(l.outputs), len(outputs))], nil
}

// nGates is the number of gates of an LSTM: input (i), output (o), forget (f) and cell
// (c), stored in that order in the weights and biases.
const nGates = 4

// nPeepholes is the number of peephole weight vectors: input, output and forget.
const nPeepholes = 3

// lstm computes all timesteps of the LSTM and returns Y, Y_h and Y_c. The input
// projections X·Wᵀ of every timestep are computed with a single matrix multiplication up
// front; each timestep then needs one multiplication with the recurrent weights.
func lstm[T ops.FloatType](
	l *LSTM, dims ops.RecurrentDims, X, W, R, B, initialH, initialC, P *tensor.Tensor,
) ([]*tensor.Tensor, error) {
	f, err := ops.ParseActivation(l.activations[0])
	if err != nil {
		return nil, err
	}

	g, err := ops.ParseActivation(l.activations[1])
	if err != nil {
		return nil, err
	}

	h, err := ops.ParseActivation(l.activations[2])
	if err != nil {
		return nil, err
	}

	hidden := l.hiddenSize
	gates := nGates * hidden
	steps := dims.SeqLength
	batch := dims.BatchSize

	xw := make([]T, steps*batch*gates)
	projectInput := ops.TransposedProduct(tensor.Data[T](W), gates, dims.InputSize, l.inputWeights.For(W))
	projectInput(xw, tensor.Data[T](X), steps*batch)

	wb, rb := ops.RecurrentBiases[T](B, nGates, hidden)
	projectHidden := ops.TransposedProduct(tensor.Data[T](R), gates, hidden, l.recurrentWeights.For(R))

	var peepholes []T
	if P != nil {
		peepholes = tensor.Data[T](P)
	}

	y := make([]T, steps*batch*hidden)
	hiddenState := ops.InitialState[T](initialH, batch*hidden)
	cellState := ops.InitialState[T](initialC, batch*hidden)
	newCellState := make([]T, batch*hidden)
	hr := make([]T, batch*gates)

	// The input and forget gates share a buffer, so one call activates both.
	inputAndForget := make([]T, 2*batch*hidden)
	inputGates, forgetGates := inputAndForget[:batch*hidden], inputAndForget[batch*hidden:]
	cellGates := make([]T, batch*hidden)
	outputGates := make([]T, batch*hidden)
	activatedCells := make([]T, batch*hidden)

	// gate returns the pre-activation value of a gate: the input and recurrent projections
	// with their biases.
	gate := func(xwRow, hrRow []T, offset, j int) T {
		return (xwRow[offset+j] + wb[offset+j]) + (hrRow[offset+j] + rb[offset+j])
	}

	for t := 0; t < steps; t++ {
		projectHidden(hr, hiddenState, batch)

		for b := 0; b < batch; b++ {
			xwRow := xw[(t*batch+b)*gates:][:gates]
			hrRow := hr[b*gates:][:gates]

			for j := 0; j < hidden; j++ {
				i := b*hidden + j

				inputGates[i] = gate(xwRow, hrRow, 0, j)
				forgetGates[i] = gate(xwRow, hrRow, 2*hidden, j)
				outputGates[i] = gate(xwRow, hrRow, hidden, j)
				cellGates[i] = gate(xwRow, hrRow, 3*hidden, j)

				if peepholes != nil {
					inputGates[i] += peepholes[j] * cellState[i]
					forgetGates[i] += peepholes[2*hidden+j] * cellState[i]
				}
			}
		}

		ops.Activate(inputAndForget, f)
		ops.Activate(cellGates, g)

		for i := range newCellState {
			cell := forgetGates[i]*cellState[i] + inputGates[i]*cellGates[i]
			newCellState[i] = cell

			if peepholes != nil {
				outputGates[i] += peepholes[hidden+i%hidden] * cell
			}
		}

		ops.Activate(outputGates, f)
		copy(activatedCells, newCellState)
		ops.Activate(activatedCells, h)

		out := y[t*batch*hidden:][:batch*hidden]
		for i := range out {
			out[i] = outputGates[i] * activatedCells[i]
		}

		hiddenState = out
		cellState, newCellState = newCellState, cellState
	}

	Y := tensor.New(y, steps, 1, batch, hidden)
	Yh := tensor.New(hiddenState, 1, batch, hidden)
	Yc := tensor.New(cellState, 1, batch, hidden)

	return []*tensor.Tensor{Y, Yh, Yc}, nil
}

// Prepare packs the weights W and R when they are parameters of the model.
func (l *LSTM) Prepare(constants []*tensor.Tensor) {
	l.inputWeights = ops.PrepareWeights(constants[1], true)
	l.recurrentWeights = ops.PrepareWeights(constants[2], true)
}
