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
	return &Conv{}
}

// Init initializes the conv operator.
func (c *Conv) Init(attributes []*onnx.AttributeProto) error {
	var err error
	for _, attr := range attributes {
		switch attr.GetName() {
		case "auto_pad":
			c.autoPad = AutoPadSetting(attr.GetS())
			if c.autoPad != "NOTSET" {
				return fmt.Errorf(ops.UnsupportedAttrErrTemplate, c, attr.GetName())
			}
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
	X := inputs[0]
	kernel := inputs[1]
	var bias tensor.Tensor = nil
	if len(inputs) == 3 {
		bias = inputs[2]
	}

	if len(c.dilations) == 0 {
		c.setDefaultDilations(X)
	}
	if len(c.kernelShape) == 0 {
		c.setKernelShape(kernel)
	}
	if len(c.pads) == 0 {
		c.setDefaultPaddings(X)
	}
	if len(c.strides) == 0 {
		c.setDefaultStrides(X)
	}

	kernel, err := c.getDilatedKernel(kernel)
	if err != nil {
		return nil, err
	}

	// 2D Convolution where
	if len(X.Shape()) == 4 {
	} else {
		return nil, fmt.Errorf("The convolution operator currently only supports 2D convolution, i.e. shape [N x C x H x W]")
	}

	return []tensor.Tensor{}, nil
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

		newCoords := c.getNewKernelCoords(oldCoords, kernel.Shape(), newKernel.Shape())
		newKernel.SetAt(value, newCoords...)
	}

	c.setKernelShape(newKernel)
	return newKernel, nil
}

func (c *Conv) getNewKernelCoords(oldCoords, oldShape, newShape []int) []int {
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
