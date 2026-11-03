package gru

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var gruTypeConstraints = [][]tensor.DataType{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Int32},
	{tensor.Float32, tensor.Float64},
}

const (
	MinGRUInputs = 3
	MaxGRUInputs = 6
)

// GRU represents the ONNX gru operator. It only supports a simple forward gru
// operation with default activations.
type GRU struct {
	ops.BaseOperator

	activationAlpha   []float32
	activationBeta    []float32
	activations       []string
	direction         ops.SequenceProcessDirection
	hiddenSize        int
	linearBeforeReset bool
}

// newGRU creates a new gru operator.
func newGRU(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &GRU{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinGRUInputs,
			MaxGRUInputs,
			typeConstraints,
			"gru",
		),
		activations:       []string{"sigmoid", "tanh"},
		direction:         ops.Forward,
		linearBeforeReset: false,
	}
}

// Init initializes the gru operator. Currently, our GRU operator does not support all
// attributes as specified by the ONNX operator. The basic functionality is working and
// the other attributes can be added later on.
func (g *GRU) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	for _, attr := range attributes {
		switch attr.GetName() {
		case ops.ActivationAlphaAttr:
			g.activationAlpha = attr.GetFloats()
		case ops.ActivationBetaAttr:
			g.activationBeta = attr.GetFloats()
		case ops.ActivationsAttr:
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			g.activations = activations
		case ops.ClipAttr:
			return ops.ErrUnsupportedAttribute(attr.GetName(), g)
		case ops.DirectionAttr:
			g.direction = ops.SequenceProcessDirection(attr.GetS())
			if g.direction != ops.Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), g)
			}
		case ops.HiddenSizeAttr:
			g.hiddenSize = int(attr.GetI())
		case "linear_before_reset":
			g.linearBeforeReset = ops.Int64ToBool(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	}

	return nil
}

// Apply applies the gru operator.
func (g *GRU) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", g.BaseOperator)
	}

	X, W, R, B, initialH := inputs[0], inputs[1], inputs[2], inputs[3], inputs[5]

	dims, err := ops.ValidateRecurrentInputs(X, W, R, B, initialH, nGates, g.hiddenSize)
	if err != nil {
		return nil, err
	}

	switch X.DataType() {
	case tensor.Float32:
		return gru[float32](g, dims, X, W, R, B, initialH)
	case tensor.Float64:
		return gru[float64](g, dims, X, W, R, B, initialH)
	default:
		return nil, ops.ErrInvalidInputType(0, X.DataType().String(), g.BaseOperator)
	}
}

// nGates is the number of gates of a GRU: update (z), reset (r) and hidden (h), stored in
// that order in the weights and biases.
const nGates = 3

// gru computes all timesteps of the GRU. The input projections X·Wᵀ of every timestep
// are computed with a single matrix multiplication up front, so only the multiplications
// with the recurrent weights remain per timestep.
func gru[T ops.FloatType](g *GRU, dims ops.RecurrentDims, X, W, R, B, initialH *tensor.Tensor) ([]*tensor.Tensor, error) {
	f, err := ops.ActivationFunc[T](g.activations[0])
	if err != nil {
		return nil, err
	}

	gActivation, err := ops.ActivationFunc[T](g.activations[1])
	if err != nil {
		return nil, err
	}

	hidden := g.hiddenSize
	gates := nGates * hidden
	steps := dims.SeqLength
	batch := dims.BatchSize

	xw := make([]T, steps*batch*gates)
	ops.MatMulInto(xw, tensor.Data[T](X), tensor.Data[T](W), steps*batch, gates, dims.InputSize, false, true)

	wb, rb := ops.RecurrentBiases[T](B, nGates, hidden)
	r := tensor.Data[T](R)
	recurrentH := r[2*hidden*hidden:]

	y := make([]T, steps*batch*hidden)
	h := ops.InitialState[T](initialH, batch*hidden)
	hr := make([]T, batch*gates)
	z := make([]T, batch*hidden)
	reset := make([]T, batch*hidden)

	var resetHidden, resetHR []T
	if !g.linearBeforeReset {
		resetHidden = make([]T, batch*hidden)
		resetHR = make([]T, batch*hidden)
	}

	for t := 0; t < steps; t++ {
		ops.MatMulInto(hr, h, r, batch, gates, hidden, false, true)

		for b := 0; b < batch; b++ {
			xwRow := xw[(t*batch+b)*gates:][:gates]
			hrRow := hr[b*gates:][:gates]

			for j := 0; j < hidden; j++ {
				z[b*hidden+j] = f((xwRow[j] + wb[j]) + (hrRow[j] + rb[j]))
				reset[b*hidden+j] = f((xwRow[hidden+j] + wb[hidden+j]) + (hrRow[hidden+j] + rb[hidden+j]))
			}
		}

		if !g.linearBeforeReset {
			for i, value := range reset {
				resetHidden[i] = value * h[i]
			}

			ops.MatMulInto(resetHR, resetHidden, recurrentH, batch, hidden, hidden, false, true)
		}

		out := y[t*batch*hidden:][:batch*hidden]

		for b := 0; b < batch; b++ {
			xwRow := xw[(t*batch+b)*gates+2*hidden:][:hidden]
			wbH, rbH := wb[2*hidden:], rb[2*hidden:]

			for j := 0; j < hidden; j++ {
				i := b*hidden + j

				var candidate T
				if g.linearBeforeReset {
					candidate = gActivation((hr[b*gates+2*hidden+j]+rbH[j])*reset[i] + (xwRow[j] + wbH[j]))
				} else {
					candidate = gActivation((xwRow[j] + wbH[j]) + (resetHR[i] + rbH[j]))
				}

				out[i] = (1-z[i])*candidate + z[i]*h[i]
			}
		}

		h = out
	}

	Y := tensor.New(y, steps, 1, batch, hidden)
	Yh := tensor.New(h, 1, batch, hidden)

	return []*tensor.Tensor{Y, Yh}, nil
}
