package gru

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/advancedclimatesystems/gonnx/ops/gemm"
	"gorgonia.org/tensor"
)

var gruTypeConstraints = [][]tensor.Dtype{
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
func newGRU(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
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
func (g *GRU) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", g.BaseOperator)
	}

	X := inputs[0]
	seqLength := X.Shape()[0]
	batchSize := X.Shape()[1]

	Wz, Wr, Wh, err := g.getWeights(inputs[1])
	if err != nil {
		return nil, err
	}

	Rz, Rr, Rh, err := g.getWeights(inputs[2])
	if err != nil {
		return nil, err
	}

	B := inputs[3]
	if B == nil {
		// 6 is the number of bias matrices required by ONNX definition.
		nBiasMatrices := 6
		B = ops.ZeroTensor(1, nBiasMatrices*g.hiddenSize)
	}

	Wbz, Wbr, Wbh, Rbz, Rbr, Rbh, err := g.getBiases(B)
	if err != nil {
		return nil, err
	}

	prevH := inputs[5]
	if prevH == nil {
		prevH = ops.ZeroTensor(1, batchSize, g.hiddenSize)
	}

	// Extract the shape of the hidden dimensions without the bidirectional dimension, as
	// we do not support bidirectional GRU yet.
	shapeWithoutBidir := prevH.Shape().Clone()[1:]

	err = prevH.Reshape(shapeWithoutBidir...)
	if err != nil {
		return nil, err
	}

	fActivation, err := ops.GetActivation(g.activations[0])
	if err != nil {
		return nil, err
	}

	gActivation, err := ops.GetActivation(g.activations[1])
	if gActivation == nil {
		return nil, err
	}

	outputs := []tensor.Tensor{}

	for i := 0; i < seqLength; i++ {
		Xt, err := g.extractXt(X, i)
		if err != nil {
			return nil, err
		}

		zt, err := g.gateCalculation(Xt, prevH, Wz, Rz, Wbz, Rbz, fActivation)
		if err != nil {
			return nil, err
		}

		rt, err := g.gateCalculation(Xt, prevH, Wr, Rr, Wbr, Rbr, fActivation)
		if err != nil {
			return nil, err
		}

		ht, err := g.htCalculation(Xt, prevH, rt, Wh, Rh, Wbh, Rbh, gActivation)
		if err != nil {
			return nil, err
		}

		prevH, err = g.hiddenCalculation(zt, ht, prevH)
		if err != nil {
			return nil, err
		}

		outputs = append(outputs, prevH)
	}

	var Y tensor.Tensor
	if len(outputs) > 1 {
		Y, err = tensor.Concat(0, outputs[0], outputs[1:]...)
		if err != nil {
			return nil, err
		}
	} else {
		Y = outputs[0]
	}

	// Reshape the output so it adds the num_directions as specified by onnx.
	err = Y.Reshape([]int{seqLength, 1, batchSize, g.hiddenSize}...)
	if err != nil {
		return nil, err
	}

	Yh, ok := prevH.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", prevH.Clone())
	}

	// Reshape the output so it adds the num_directions as specified by onnx.
	err = Yh.Reshape([]int{1, batchSize, g.hiddenSize}...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{Y, Yh}, nil
}

// extractXt extracts the value of x for timestep t.
func (g *GRU) extractXt(X tensor.Tensor, t int) (tensor.Tensor, error) {
	return X.Slice(ops.NewSlicer(t, t+1), nil, nil)
}

func (g *GRU) gateCalculation(
	Xt, H, W, R, Wb, Rb tensor.Tensor, activation ops.Activation,
) (tensor.Tensor, error) {
	gemm := gemm.GetVersions()[13]()

	err := gemm.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "alpha", F: 1.0},
				{Name: "beta", F: 1.0},
				{Name: "transA", I: 0},
				{Name: "transB", I: 1},
			},
		},
	)
	if err != nil {
		return nil, err
	}

	inputCalc, err := gemm.Apply([]tensor.Tensor{Xt, W, Wb})
	if err != nil {
		return nil, err
	}

	hiddenCalc, err := gemm.Apply([]tensor.Tensor{H, R, Rb})
	if err != nil {
		return nil, err
	}

	gate, err := tensor.Add(inputCalc[0], hiddenCalc[0])
	if err != nil {
		return nil, err
	}

	return activation(gate)
}

func (g *GRU) htCalculation(
	Xt, prevH, rt, W, R, Wb, Rb tensor.Tensor, activation ops.Activation,
) (tensor.Tensor, error) {
	if !g.linearBeforeReset {
		temp1, err := tensor.Mul(rt, prevH)
		if err != nil {
			return nil, err
		}

		return g.gateCalculation(Xt, temp1, W, R, Wb, Rb, activation)
	}

	gemm := gemm.GetVersions()[13]()

	err := gemm.Init(
		&onnx.NodeProto{
			Attribute: []*onnx.AttributeProto{
				{Name: "alpha", F: 1.0},
				{Name: "beta", F: 1.0},
				{Name: "transA", I: 0},
				{Name: "transB", I: 1},
			},
		},
	)
	if err != nil {
		return nil, err
	}

	inputCalc, err := gemm.Apply([]tensor.Tensor{Xt, W, Wb})
	if err != nil {
		return nil, err
	}

	hiddenCalc, err := gemm.Apply([]tensor.Tensor{prevH, R, Rb})
	if err != nil {
		return nil, err
	}

	temp1, err := tensor.Mul(hiddenCalc[0], rt)
	if err != nil {
		return nil, err
	}

	temp2, err := tensor.Add(temp1, inputCalc[0])
	if err != nil {
		return nil, err
	}

	return activation(temp2)
}

func (g *GRU) hiddenCalculation(zt, ht, prevH tensor.Tensor) (tensor.Tensor, error) {
	temp1, err := tensor.Sub(ops.OnesTensor(zt), zt)
	if err != nil {
		return nil, err
	}

	temp2, err := tensor.Mul(temp1, ht)
	if err != nil {
		return nil, err
	}

	temp3, err := tensor.Mul(zt, prevH)
	if err != nil {
		return nil, err
	}

	return tensor.Add(temp2, temp3)
}

// getWeights splits tensor W into 3 weight matrices.
// The W tensor, by GONNX definition, has 3 dimensions with 3 weight
// tensors in it (6 if bidirectional, but that is not supported).
func (g *GRU) getWeights(W tensor.Tensor) (Wz, Wr, Wh tensor.Tensor, err error) {
	nWeightMatrices := 3
	nWeightDimensions := 3

	weights, err := ops.ExtractMatrices(W, nWeightMatrices, nWeightDimensions, g.hiddenSize)
	if err != nil {
		return nil, nil, nil, err
	}

	return weights[0], weights[1], weights[2], nil
}

// getBiases returns the biases from the Bias node as specified by the ONNX standard.
// The B tensor, by GONNX definition, has 2 dimensions with 6 bias
// tensors in it (12 if bidirectional, but that is not supported).
func (g *GRU) getBiases(B tensor.Tensor) (Wbz, Wbr, Wbh, Rbz, Rbr, Rbh tensor.Tensor, err error) {
	nBiasMatrices := 6
	nBiasDimensions := 2

	biases, err := ops.ExtractMatrices(B, nBiasMatrices, nBiasDimensions, g.hiddenSize)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	return biases[0], biases[1], biases[2], biases[3], biases[4], biases[5], nil
}
