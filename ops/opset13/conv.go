package opset13

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

type AutoPadSetting string

const (
	NotSet    AutoPadSetting = "NOTSET"
	SameUpper AutoPadSetting = "SAME_UPPER"
	SameLower AutoPadSetting = "SAME_LOWER"
	Valid     AutoPadSetting = "VALID"
)

// The number of non spatial dimensions inputs and kernels will always have.
// For input tensors, the first dimension will be the batch size.
// For kernel tensors, the first dimension will be the number of kernels.
// For all tensors, the second dimension will be the number of channels.
const nNonSpatialDims = 2

// Conv represents the ONNX conv operator.
type Conv struct {
	// Type of padding to apply before doing the convolutions.
	autoPad AutoPadSetting

	// Dilation value along each dimension of the filter.
	dilations []int

	// Numer of groups the input channels and the output channels are divided into.
	group int

	// Shape of the convolutional kernel. Can be present, but if not should be inferred (i.e. useless attribute).
	kernelShape []int

	// Padding for the beginning and ending of each dimension. Cannot be used with autopad setting.
	pads []int

	// Strides along each dimension.
	strides []int
}

// newConv creates a new conv operator.
func newConv() ops.Operator {
	return &Conv{
		autoPad: "NOTSET",
	}
}

// Init initializes the conv operator.
func (c *Conv) Init(attributes []*onnx.AttributeProto) error {
	var err error

	for _, attr := range attributes {
		switch attr.GetName() {
		case "auto_pad":
			c.autoPad = AutoPadSetting(attr.GetS())
		case "dilations":
			c.dilations, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return fmt.Errorf(ops.InvalidAttrTemplate, c, attr.GetName(), c.dilations)
			}
		case "group":
			c.group = int(attr.GetI())
			if c.group != 1 {
				return fmt.Errorf(ops.UnsupportedAttrErrTemplate, c, attr.GetName())
			}
		case "kernel_shape":
			c.kernelShape, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return fmt.Errorf(ops.InvalidAttrTemplate, c, attr.GetName(), c.kernelShape)
			}
		case "pads":
			c.pads, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return fmt.Errorf(ops.InvalidAttrTemplate, c, attr.GetName(), c.pads)
			}
		case "strides":
			c.strides, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return fmt.Errorf(ops.InvalidAttrTemplate, c, attr.GetName(), c.strides)
			}
		default:
			return fmt.Errorf(ops.UnsupportedAttrErrTemplate, c, attr.GetName())
		}
	}

	return nil
}

// Apply applies the conv operator.
func (c *Conv) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0]
	kernel := inputs[1]

	var bias tensor.Tensor
	if len(inputs) == c.GetMaxInputs() {
		bias = inputs[2]
	}

	if len(c.dilations) == 0 {
		c.setDefaultDilations(x)
	}

	if len(c.kernelShape) == 0 {
		c.setKernelShape(kernel)
	}

	if len(c.pads) == 0 {
		c.setDefaultPaddings(x)
	}

	if len(c.strides) == 0 {
		c.setDefaultStrides(x)
	}

	kernel, err := c.getDilatedKernel(kernel)
	if err != nil {
		return nil, err
	}

	if c.autoPad != "NOTSET" {
		c.setPaddingWithAutoPad(x)
	}

	var out tensor.Tensor

	switch len(x.Shape()) {
	case 3:
		out, err = c.applyConv1D(x, kernel)
	case 4:
		out, err = c.applyConv2D(x, kernel)
	default:
		return nil, fmt.Errorf("The convolution operator currently only supports 1D or 2D convolution, i.e. shape [N x C x H (x W)]")
	}

	if err != nil {
		return nil, err
	}

	if bias != nil {
		out, err = c.addBias(out, bias)
		if err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{out}, nil
}

// ValidateInputs validates the inputs that will be given to Apply for this operator.
func (c *Conv) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return ops.ValidateInputs(c, inputs)
}

// GetMinInputs returns the minimum number of input tensors this operator expects.
func (c *Conv) GetMinInputs() int {
	return 2
}

// GetMaxInputs returns the maximum number of input tensors this operator expects.
func (c *Conv) GetMaxInputs() int {
	return 3
}

// GetInputTypeConstraints returns a list. Every element represents a set of allowed tensor dtypes
// for the corresponding input tensor.
func (c *Conv) GetInputTypeConstraints() [][]tensor.Dtype {
	return [][]tensor.Dtype{
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
		{tensor.Float32, tensor.Float64},
	}
}

// String implements the stringer interface, and can be used to format errors or messages.
func (c *Conv) String() string {
	return "conv operator"
}

// setDefaultDilations sets the dilations attribute to the default. Can be called when no
// dilations were set when initializing.
func (c *Conv) setDefaultDilations(x tensor.Tensor) {
	nDims := len(x.Shape()[2:])

	dilations := make([]int, nDims)
	for i := 0; i < nDims; i++ {
		dilations[i] = 1
	}

	c.dilations = dilations
}

// setKernelShape infers the shape of the kernel when it was not given in the attributes.
func (c *Conv) setKernelShape(kernel tensor.Tensor) {
	c.kernelShape = kernel.Shape()[2:]
}

// setDefaultPaddings sets default paddings as attribute. Can be called when no paddings
// were set during initialization.
func (c *Conv) setDefaultPaddings(x tensor.Tensor) {
	paddingLength := len(x.Shape()[2:]) * 2

	pads := make([]int, paddingLength)
	for i := 0; i < paddingLength; i++ {
		pads[i] = 0
	}

	c.pads = pads
}

// setDefaultStrides sets default strides as attribute. Can be called when no strides
// were set during initialization.
func (c *Conv) setDefaultStrides(x tensor.Tensor) {
	nDims := len(x.Shape()[2:])

	strides := make([]int, nDims)
	for i := 0; i < nDims; i++ {
		strides[i] = 1
	}

	c.strides = strides
}

// setPaddingWithAutoPad sets the padding attribute of the operator based on
// the input tensor `x`, the shape of the kernel and the strides.
func (c *Conv) setPaddingWithAutoPad(x tensor.Tensor) {
	if c.autoPad == "NOTSET" {
		return
	}

	inputShape := x.Shape()
	nDims := len(inputShape)
	nSpatialDims := nDims - nNonSpatialDims

	c.pads = make([]int, nSpatialDims*2)

	for i := 0; i < nSpatialDims; i++ {
		dim := inputShape[i]
		targetSize := (dim + c.strides[i] - 1) / c.strides[i]
		padNeeded := (targetSize-1)*c.strides[i] + c.kernelShape[i] - dim

		var padHead int
		if c.autoPad == "SAME_LOWER" {
			padHead = (padNeeded + 1) / 2
		} else {
			padHead = padNeeded / 2
		}

		padTail := padNeeded - padHead
		c.pads[i] = padHead
		c.pads[i+nSpatialDims] = padTail
	}
}

// getDilatedKernel creates a new kernel given the `dilations` attribute of this
// conv operator. A dilated kernel basically means inserting zeros in between
// the kernels, i.e. a 2D kernel like:
//
//	1 2
//	3 4
//
// Dilated by one in both dimensions yields a new kernel of:
//
//	1 0 2
//	0 0 0
//	3 0 4
//
// This function updates the given kernel and dilates it by the given amount
// for each dimensions separately. It returns a new tensor with the new kernel.
func (c *Conv) getDilatedKernel(kernel tensor.Tensor) (tensor.Tensor, error) {
	oldKernelShape := kernel.Shape()
	newKernelShape := make([]int, len(oldKernelShape))

	// Add the non spatial dimensions of the kernel, i.e. the number of
	// kernels (index 0) and the number of channels (index 1). These
	// dimensions do not have to be dilated.
	for i := 0; i < nNonSpatialDims; i++ {
		newKernelShape[i] = oldKernelShape[i]
	}

	// Add the dilated spatial dimensions of the kernel, i.e. in the case
	// of 2D images these are the width and height dimensions.
	for i, dilation := range c.dilations {
		oldKernelDim := oldKernelShape[nNonSpatialDims+i]
		newKernelShape[nNonSpatialDims+i] = oldKernelDim + (oldKernelDim-1)*(dilation-1)
	}

	newKernel := tensor.NewDense(kernel.Dtype(), newKernelShape)
	newKernel.Zero()

	// Now we fill the empty kernel with the original kernel values at the
	// right positions.
	iterator := kernel.Iterator()
	iterator.Reset()

	for !iterator.Done() {
		oldCoords := iterator.Coord()

		value, err := kernel.At(oldCoords...)
		if err != nil {
			return nil, err
		}

		newCoords := c.getNewCoordsAfterDilation(oldCoords)

		err = newKernel.SetAt(value, newCoords...)
		if err != nil {
			return nil, err
		}

		_, err = iterator.Next()
		if err != nil {
			return nil, err
		}
	}

	c.setKernelShape(newKernel)

	return newKernel, nil
}

// getNewCoordsAfterDilation returns the new coordinates of a value given the old coordinates of that
// value in the old kernel and its shape. The new coordinates can be used to store the value/weight
// in the dilated kernel.
func (c *Conv) getNewCoordsAfterDilation(oldCoords []int) []int {
	newCoords := make([]int, len(oldCoords))

	for i := 0; i < nNonSpatialDims; i++ {
		newCoords[i] = oldCoords[i]
	}

	for i, dilation := range c.dilations {
		newCoords[nNonSpatialDims+i] = oldCoords[nNonSpatialDims+i] * dilation
	}

	return newCoords
}

// Applies 1D convolution to tensor X with the 'kernel' tensor.
// X will have 3 dimensions: [N, C, H] where N is the batch size, C is the number
// of channels and H is the number of dimensions on which to apply the convolutions.
// The kernel will have shape [kernelDim], where 'kernelDim' is the size of the kernel
// size of the kernel.
func (c *Conv) applyConv1D(x, kernel tensor.Tensor) (tensor.Tensor, error) {
	outputShape := c.getOutputShape(x, kernel)
	out := tensor.Tensor(tensor.NewDense(x.Dtype(), outputShape))
	out.Zero()

	paddedX, err := c.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	nKernels := kernel.Shape()[0]
	strideSize := c.strides[0]
	outputHDim := outputShape[nNonSpatialDims]

	for batchIdx := 0; batchIdx < nBatches; batchIdx++ {
		for kernelIdx := 0; kernelIdx < nKernels; kernelIdx++ {
			subKernelView, err := kernel.Slice(ops.NewSlicer(kernelIdx, kernelIdx+1))
			if err != nil {
				return nil, err
			}

			subKernel := subKernelView.Materialize()

			for h := 0; h < paddedX.Shape()[2]; h += strideSize {
				dimHOutputIdx := h / strideSize
				if dimHOutputIdx >= outputHDim {
					continue
				}

				subImage, err := c.getSubImage(paddedX, batchIdx, h)
				if err != nil {
					return nil, err
				}

				subImage, subKernel, err = ops.UnidirectionalBroadcast(subImage, subKernel)
				if err != nil {
					return nil, err
				}

				convResult, err := tensor.Mul(subImage, subKernel)
				if err != nil {
					return nil, err
				}

				convValue, err := tensor.Sum(convResult)
				if err != nil {
					return nil, err
				}

				err = out.SetAt(convValue.ScalarValue(), batchIdx, kernelIdx, dimHOutputIdx)
				if err != nil {
					return nil, err
				}
			}
		}
	}

	return out, nil
}

// Applies 2D convolution to tensor X with the 'kernel' tensor.
// X will have 4 dimensions: [N, C, H, W] where N is the batch size, C is the number
// of channels, H and W are the height and width dimensions on which to apply the convolutions.
// The kernel will have shape [M, C, H, W].
func (c *Conv) applyConv2D(x, kernel tensor.Tensor) (tensor.Tensor, error) {
	outputShape := c.getOutputShape(x, kernel)
	out := tensor.Tensor(tensor.NewDense(x.Dtype(), outputShape))
	out.Zero()

	outputHDim := outputShape[nNonSpatialDims]
	outputWDim := outputShape[nNonSpatialDims+1]

	paddedX, err := c.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	nKernels := kernel.Shape()[0]

	for batchIdx := 0; batchIdx < nBatches; batchIdx++ {
		for kernelIdx := 0; kernelIdx < nKernels; kernelIdx++ {
			subKernelView, err := kernel.Slice(ops.NewSlicer(kernelIdx, kernelIdx+1))
			if err != nil {
				return nil, err
			}

			subKernel := subKernelView.Materialize()

			// Loop over all 2D subImages of the input image and compute the convolution
			// for that subImage. Store the result at the right place in the output tensor.
			for h := 0; h < paddedX.Shape()[2]; h += c.strides[0] {
				dimHOutputIdx := h / c.strides[0]
				if dimHOutputIdx >= outputHDim {
					continue
				}

				for w := 0; w < paddedX.Shape()[2]; w += c.strides[1] {
					dimWOutputIdx := w / c.strides[1]
					if dimWOutputIdx >= outputWDim {
						continue
					}

					subImage, err := c.getSubImage(paddedX, batchIdx, h, w)
					if err != nil {
						return nil, err
					}

					subImage, subKernel, err = ops.UnidirectionalBroadcast(subImage, subKernel)
					if err != nil {
						return nil, err
					}

					convResult, err := tensor.Mul(subImage, subKernel)
					if err != nil {
						return nil, err
					}

					convValue, err := tensor.Sum(convResult)
					if err != nil {
						return nil, err
					}

					err = out.SetAt(convValue.ScalarValue(), batchIdx, kernelIdx, dimHOutputIdx, dimWOutputIdx)
					if err != nil {
						return nil, err
					}
				}
			}
		}
	}

	return out, nil
}

// getOutputShape calculates the shape of the output tensor resulting from
// the convolution operation between `x` and `kernel`.
// `x` has shape [N, C, H, W, ...] and `kernel` has shape [M, C, H, W, ...].
// The output shape will be [N, M, newH, newW, ...], where values like `newH`
// are calculated based on the input shape, kernel size, padding and strides.
func (c *Conv) getOutputShape(x, kernel tensor.Tensor) tensor.Shape {
	outputShape := make([]int, len(x.Shape()))

	outputShape[0] = x.Shape()[0]
	outputShape[1] = kernel.Shape()[0]

	nSpatialDims := len(x.Shape()) - nNonSpatialDims
	for i := 0; i < nSpatialDims; i++ {
		inputDim := x.Shape()[nNonSpatialDims+i]
		kernelDim := c.kernelShape[i]
		outputShape[nNonSpatialDims+i] = ((inputDim - kernelDim + c.pads[i] + c.pads[i+nSpatialDims]) / c.strides[i]) + 1
	}

	return outputShape
}

// padInput pads the input with zeros according to the `pads` attribute.
// The pad attribute specifies how many zeros should be added before and
// after the values in that specific dimension.
// Please note that according to ONNX specs, the `pads` attributes is an
// array with pads as [x1_begin, x2_begin, ..., x1_after, x2_after].
// This method achieves padding by concatting tensors with zero values
// before and after each spatial dimension of the input tensor `x`.
func (c *Conv) padInput(x tensor.Tensor) (tensor.Tensor, error) {
	var err error

	nSpatialDims := len(x.Shape()[nNonSpatialDims:])

	for i := 0; i < nSpatialDims; i++ {
		if c.pads[i] != 0 {
			padsBeforeShape := x.Shape().Clone()
			padsBeforeShape[nNonSpatialDims+i] = c.pads[i]
			zerosBefore := tensor.Tensor(tensor.NewDense(x.Dtype(), padsBeforeShape))
			zerosBefore.Zero()

			x, err = tensor.Concat(nNonSpatialDims+i, zerosBefore, x)
			if err != nil {
				return nil, err
			}
		}

		if c.pads[i+nSpatialDims] != 0 {
			padsAfterShape := x.Shape().Clone()
			padsAfterShape[nNonSpatialDims+i] = c.pads[i+nSpatialDims]
			zerosAfter := tensor.Tensor(tensor.NewDense(x.Dtype(), padsAfterShape))
			zerosAfter.Zero()

			x, err = tensor.Concat(nNonSpatialDims+i, x, zerosAfter)
			if err != nil {
				return nil, err
			}
		}
	}

	return x, nil
}

// getSubImage returns a the subimage for a specific example in the batch, based on the
// kernel shape and the given start coordinates. The resulting sub image will be of
// shape [1, C, kernelShape[0], kernelShape[1], ...].
func (c *Conv) getSubImage(x tensor.Tensor, batchIdx int, startSpatialCoords ...int) (tensor.Tensor, error) {
	if len(startSpatialCoords) != len(c.kernelShape) {
		return nil, fmt.Errorf("expected the coordinates to have the same number of dimensions as the kernel")
	}

	slices := []tensor.Slice{
		ops.NewSlicer(batchIdx, batchIdx+1),
		nil, // Take all channels at once.
	}

	for i := 0; i < len(c.kernelShape); i++ {
		dimStartIdx := startSpatialCoords[i]
		dimKernelSize := c.kernelShape[i]
		slices = append(slices, ops.NewSlicer(dimStartIdx, dimStartIdx+dimKernelSize))
	}

	subImage, err := x.Slice(slices...)
	if err != nil {
		return nil, err
	}

	return subImage.Materialize(), nil
}

// addBias adds a bias to the output of the convolution. It reshapes the
// bias such that it can be broadcasted, and then is added to the output
// tensor.
func (c *Conv) addBias(out, bias tensor.Tensor) (tensor.Tensor, error) {
	biasShape := make([]int, len(out.Shape()))
	for i := 0; i < len(out.Shape()); i++ {
		biasShape[i] = 1
	}

	biasShape[1] = bias.Shape()[0]

	err := bias.Reshape(biasShape...)
	if err != nil {
		return nil, err
	}

	out, bias, err = ops.UnidirectionalBroadcast(out, bias)
	if err != nil {
		return nil, err
	}

	return tensor.Add(out, bias)
}
