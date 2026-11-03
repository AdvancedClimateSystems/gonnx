package rnn

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var rnnTypeConstraints = [][]tensor.DataType{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Int32},
	{tensor.Float32, tensor.Float64},
}

const (
	MinRNNInputs = 3
	MaxRNNInputs = 6
)

// RNN represents the ONNX rnn operator.
type RNN struct {
	ops.BaseOperator

	activationAlpha []float32
	activationBeta  []float32
	activations     []string
	direction       ops.SequenceProcessDirection
	hiddenSize      int

	inputWeights     *ops.PreparedWeights
	recurrentWeights *ops.PreparedWeights
}

// newRNN creates a new rnn operator.
func newRNN(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &RNN{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinRNNInputs,
			MaxRNNInputs,
			typeConstraints,
			"rnn",
		),
		activations: []string{"tanh"},
		direction:   ops.Forward,
	}
}

// Init initializes the rnn operator.
func (r *RNN) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case ops.ActivationAlphaAttr:
			r.activationAlpha = attr.GetFloats()
		case ops.ActivationBetaAttr:
			r.activationBeta = attr.GetFloats()
		case ops.ActivationsAttr:
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			r.activations = activations
		case ops.ClipAttr:
			return ops.ErrUnsupportedAttribute(attr.GetName(), r)
		case ops.DirectionAttr:
			r.direction = ops.SequenceProcessDirection(attr.GetS())
			if r.direction != ops.Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), r)
			}
		case ops.HiddenSizeAttr:
			r.hiddenSize = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

// Apply applies the rnn operator.
func (r *RNN) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", r.BaseOperator)
	}

	X, W, R, B, initialH := inputs[0], inputs[1], inputs[2], inputs[3], inputs[5]

	dims, err := ops.ValidateRecurrentInputs(X, W, R, B, initialH, 1, r.hiddenSize)
	if err != nil {
		return nil, err
	}

	switch X.DataType() {
	case tensor.Float32:
		return rnn[float32](r, dims, X, W, R, B, initialH)
	case tensor.Float64:
		return rnn[float64](r, dims, X, W, R, B, initialH)
	default:
		return nil, ops.ErrInvalidInputType(0, X.DataType().String(), r.BaseOperator)
	}
}

// rnn computes all timesteps of the RNN. The input projections X·Wᵀ of every timestep
// are computed with a single matrix multiplication up front; each timestep then needs one
// multiplication with the recurrent weights.
func rnn[T ops.FloatType](r *RNN, dims ops.RecurrentDims, X, W, R, B, initialH *tensor.Tensor) ([]*tensor.Tensor, error) {
	f, err := ops.ParseActivation(r.activations[0])
	if err != nil {
		return nil, err
	}

	hidden := r.hiddenSize
	steps := dims.SeqLength
	batch := dims.BatchSize

	xw := make([]T, steps*batch*hidden)
	projectInput := ops.TransposedProduct(tensor.Data[T](W), hidden, dims.InputSize, r.inputWeights.For(W))
	projectInput(xw, tensor.Data[T](X), steps*batch)

	wb, rb := ops.RecurrentBiases[T](B, 1, hidden)
	projectHidden := ops.TransposedProduct(tensor.Data[T](R), hidden, hidden, r.recurrentWeights.For(R))

	y := make([]T, steps*batch*hidden)
	h := ops.InitialState[T](initialH, batch*hidden)
	hr := make([]T, batch*hidden)

	for t := 0; t < steps; t++ {
		projectHidden(hr, h, batch)

		out := y[t*batch*hidden:][:batch*hidden]
		xwStep := xw[t*batch*hidden:][:batch*hidden]

		for i := range out {
			j := i % hidden
			out[i] = (xwStep[i] + wb[j]) + (hr[i] + rb[j])
		}

		ops.Activate(out, f)

		h = out
	}

	Y := tensor.New(y, steps, 1, batch, hidden)
	Yh := tensor.New(h, 1, batch, hidden)

	return []*tensor.Tensor{Y, Yh}, nil
}

// Prepare packs the weights W and R when they are parameters of the model.
func (r *RNN) Prepare(constants []*tensor.Tensor) {
	r.inputWeights = ops.PrepareWeights(constants[1], true)
	r.recurrentWeights = ops.PrepareWeights(constants[2], true)
}
