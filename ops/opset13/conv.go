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

	if len(inputs) == 3 {
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
	if len(x.Shape()) == 3 {
		out, err = c.applyConv1D(x, kernel, bias)
	} else if len(x.Shape()) == 4 {
		out, err = c.applyConv2D(x, kernel, bias)
	} else {
		return nil, fmt.Errorf("The convolution operator currently only supports 1D or 2D convolution, i.e. shape [N x C x H (x W)]")
	}

	if err != nil {
		return nil, err
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
func (c *Conv) setDefaultDilations(X tensor.Tensor) {
	nDims := len(X.Shape()[2:])

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
func (c *Conv) setDefaultPaddings(X tensor.Tensor) {
	paddingLength := len(X.Shape()[2:]) * 2

	pads := make([]int, paddingLength)
	for i := 0; i < paddingLength; i++ {
		pads[i] = 0
	}

	c.pads = pads
}

// setDefaultStrides sets default strides as attribute. Can be called when no strides
// were set during initialization.
func (c *Conv) setDefaultStrides(X tensor.Tensor) {
	nDims := len(X.Shape()[2:])

	strides := make([]int, nDims)
	for i := 0; i < nDims; i++ {
		strides[i] = 1
	}

	c.strides = strides
}

func (c *Conv) setPaddingWithAutoPad(x tensor.Tensor) {
	if c.autoPad == "NOTSET" {
		return
	}

	inputShape := x.Shape()
	nDims := len(inputShape)
	nNonSpatialDims := 2
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
	nNonSpatialDims := 2
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
	for iterator.Reset(); !iterator.Done(); iterator.Next() {
		oldCoords := iterator.Coord()
		value, err := kernel.At(oldCoords...)
		if err != nil {
			return nil, err
		}

		newCoords := c.getNewCoordsAfterDilation(oldCoords, kernel.Shape())
		newKernel.SetAt(value, newCoords...)
	}

	c.setKernelShape(newKernel)
	return newKernel, nil
}

// getNewCoordsAfterDilation returns the new coordinates of a value given the old coordinates of that
// value in the old kernel and its shape. The new coordinates can be used to store the value/weight
// in the dilated kernel.
func (c *Conv) getNewCoordsAfterDilation(oldCoords, oldShape []int) []int {
	newCoords := make([]int, len(oldCoords))

	nNonSpatialDims := 2
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
func (c *Conv) applyConv1D(x, kernel, bias tensor.Tensor) (tensor.Tensor, error) {
	dimH := x.Shape()[2]
	kernelSize := c.kernelShape[0]
	strideSize := c.strides[0]

	outputDim := ((dimH - kernelSize + c.pads[0] + c.pads[1]) / strideSize) + 1
	outputShape := []int{x.Shape()[0], kernel.Shape()[0], outputDim}
	out := tensor.Tensor(tensor.NewDense(x.Dtype(), outputShape))
	out.Zero()

	if bias != nil {
		err := bias.Reshape(1, bias.Shape()[0], 1)
		if err != nil {
			return nil, err
		}

		out, bias, err = ops.MultidirectionalBroadcast(out, bias)
		if err != nil {
			return nil, err
		}

		out, err = tensor.Add(out, bias)
		if err != nil {
			return nil, err
		}
	}

	paddedX, err := c.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	nKernels := kernel.Shape()[0]

	for batchIdx := 0; batchIdx < nBatches; batchIdx++ {
		for kernelIdx := 0; kernelIdx < nKernels; kernelIdx++ {
			subKernel, err := kernel.Slice(ops.NewSlicer(kernelIdx, kernelIdx+1))
			if err != nil {
				return nil, err
			}

			for i := 0; i < paddedX.Shape()[2]; i += strideSize {
				subImage, err := c.getSubImage(paddedX, batchIdx, i)
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

				dimOutputIdx := i / strideSize

				err = out.SetAt(convValue.ScalarValue(), batchIdx, kernelIdx, dimOutputIdx)
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
func (c *Conv) applyConv2D(x, kernel, bias tensor.Tensor) (tensor.Tensor, error) {
	dimH := x.Shape()[2]
	dimW := x.Shape()[3]

	kernelHSize := c.kernelShape[0]
	kernelWSize := c.kernelShape[1]
	strideHSize := c.strides[0]
	strideWSize := c.strides[1]

	outputHDim := ((dimH - kernelHSize + c.pads[0] + c.pads[2]) / strideHSize) + 1
	outputWDim := ((dimW - kernelWSize + c.pads[1] + c.pads[3]) / strideWSize) + 1
	outputShape := []int{x.Shape()[0], kernel.Shape()[0], outputHDim, outputWDim}
	out := tensor.Tensor(tensor.NewDense(x.Dtype(), outputShape))
	out.Zero()

	paddedX, err := c.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	nKernels := kernel.Shape()[0]

	for batchIdx := 0; batchIdx < nBatches; batchIdx++ {
		for kernelIdx := 0; kernelIdx < nKernels; kernelIdx++ {
			subKernel, err := kernel.Slice(ops.NewSlicer(kernelIdx, kernelIdx+1))
			if err != nil {
				return nil, err
			}

			for h := 0; h < paddedX.Shape()[2]; h += strideHSize {
				dimHOutputIdx := h / strideHSize
				if dimHOutputIdx >= outputHDim {
					continue
				}

				for w := 0; w < paddedX.Shape()[2]; w += strideWSize {
					dimWOutputIdx := w / strideWSize
					if dimWOutputIdx >= outputWDim {
						continue
					}

					subImage, err := c.getSubImage(paddedX, batchIdx, h, w)
					if err != nil {
						return nil, err
					}

					copiedSubKernel := subKernel.Materialize()
					copiedSubImage := subImage.Materialize()

					convResult, err := tensor.Mul(copiedSubImage, copiedSubKernel)
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

	if bias != nil {
		err := bias.Reshape(1, bias.Shape()[0], 1, 1)
		if err != nil {
			return nil, err
		}

		out, bias, err = ops.MultidirectionalBroadcast(out, bias)
		if err != nil {
			return nil, err
		}

		out, err = tensor.Add(out, bias)
		if err != nil {
			return nil, err
		}
	}

	return out, nil
}

func (c *Conv) padInput(x tensor.Tensor) (tensor.Tensor, error) {
	var err error

	nSpatialDims := len(x.Shape()[2:])
	nNonSpatialDims := 2

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
func (c *Conv) getSubImage(x tensor.Tensor, batchIdx int, startSpatialCoords ...int) (tensor.View, error) {
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

	return x.Slice(slices...)
}
