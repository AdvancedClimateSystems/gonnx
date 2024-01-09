package opset13

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

const (
	MinBatchNormalizationInputs       = 5
	MaxBatchNormalizationInputs       = 5
	BatchNormalizationDefaultEpsilon  = 1e-5
	BatchNormalizationDefaultMomentum = 0.9
)

// BatchNormalization represents the ONNX batchNormalization operator.
type BatchNormalization struct {
	epsilon  float32
	momentum float32
	testMode bool
}

// newBatchNormalization creates a new batchNormalization operator.
func newBatchNormalization() ops.Operator {
	return &BatchNormalization{
		epsilon:  BatchNormalizationDefaultEpsilon,
		momentum: BatchNormalizationDefaultMomentum,
	}
}

// Init initializes the batchNormalization operator.
func (b *BatchNormalization) Init(n *onnx.NodeProto) error {
	hasMomentum := false

	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "epsilon":
			b.epsilon = attr.GetF()
		case "momentum":
			hasMomentum = true
			b.momentum = attr.GetF()
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), b)
		}
	}

	if !hasMomentum {
		b.testMode = true
	}

	return nil
}

// Apply applies the batchNormalization operator.
func (b *BatchNormalization) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	X := inputs[0]
	scale := inputs[1]
	B := inputs[2]
	mean := inputs[3]
	variance := inputs[4]

	// We only support test mode, as this is by far the most common for inference models.
	if !b.testMode {
		return nil, ops.ErrUnsupportedAttribute("momentum", b)
	}

	out, err := b.testModeCalculation(X, scale, B, mean, variance)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (b *BatchNormalization) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(b, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (b *BatchNormalization) GetMinInputs() int {
	return MinBatchNormalizationInputs
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (b *BatchNormalization) GetMaxInputs() int {
	return MaxBatchNormalizationInputs
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (b *BatchNormalization) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (b *BatchNormalization) String() string {
	return "batchNormalization operator"
}

func (b *BatchNormalization) reshapeTensors(X, scale, bias, mean, variance tensor.Tensor) (newScale, newBias, newMean, newVariance tensor.Tensor, err error) {
	nNonSpatialDims := 2

	nSpatialDims := len(X.Shape()) - nNonSpatialDims
	if nSpatialDims <= 0 {
		return scale, bias, mean, variance, nil
	}

	// The new shape for the `scale`, `bias`, `mean` and `variance` tensors should
	// be (C, 1, 1, ...), such that they can be broadcasted to match the shape of `X`.
	newShape := make([]int, 1+nSpatialDims)

	// Here we set the channel dimension. The channel dimension is the same
	// for all `X`, `scale`, `bias`, `mean` and `variance` tensors.
	newShape[0] = scale.Shape()[0]

	// Set all the remaining dimensions to 1 to allow for broadcasting.
	for i := 1; i < len(newShape); i++ {
		newShape[i] = 1
	}

	// Now we create new tensors for all the input tensors (except `X`) and reshape
	// them.
	newScale, ok := scale.Clone().(tensor.Tensor)
	if !ok {
		return nil, nil, nil, nil, ops.ErrTypeAssert("tensor.Tensor", scale.Clone())
	}

	newBias, ok = bias.Clone().(tensor.Tensor)
	if !ok {
		return nil, nil, nil, nil, ops.ErrTypeAssert("tensor.Tensor", bias.Clone())
	}

	newMean, ok = mean.Clone().(tensor.Tensor)
	if !ok {
		return nil, nil, nil, nil, ops.ErrTypeAssert("tensor.Tensor", mean.Clone())
	}

	newVariance, ok = variance.Clone().(tensor.Tensor)
	if !ok {
		return nil, nil, nil, nil, ops.ErrTypeAssert("tensor.Tensor", variance.Clone())
	}

	err = newScale.Reshape(newShape...)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	err = newBias.Reshape(newShape...)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	err = newMean.Reshape(newShape...)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	err = newVariance.Reshape(newShape...)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return
}

func (b *BatchNormalization) testModeCalculation(X, scale, bias, mean, variance tensor.Tensor) (tensor.Tensor, error) {
	newScale, newBias, newMean, newVariance, err := b.reshapeTensors(X, scale, bias, mean, variance)
	if err != nil {
		return nil, err
	}

	numerator, err := ops.ApplyBinaryOperation(
		X,
		newMean,
		ops.Sub,
		ops.UnidirectionalBroadcasting,
	)
	if err != nil {
		return nil, err
	}

	numerator, err = ops.ApplyBinaryOperation(
		numerator[0],
		newScale,
		ops.Mul,
		ops.UnidirectionalBroadcasting,
	)
	if err != nil {
		return nil, err
	}

	denominator, err := tensor.Add(newVariance, b.epsilon)
	if err != nil {
		return nil, err
	}

	denominator, err = tensor.Sqrt(denominator)
	if err != nil {
		return nil, err
	}

	outputs, err := ops.ApplyBinaryOperation(
		numerator[0],
		denominator,
		ops.Div,
		ops.UnidirectionalBroadcasting,
	)
	if err != nil {
		return nil, err
	}

	outputs, err = ops.ApplyBinaryOperation(
		outputs[0],
		newBias,
		ops.Add,
		ops.UnidirectionalBroadcasting,
	)
	if err != nil {
		return nil, err
	}

	return outputs[0], nil
}
