package conv

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/tensor"
)

var convTypeConstraints = [][]tensor.DataType{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

var (
	MinConvInputs      = 2
	MaxConvInputs      = 3
	NDims1DConvolution = 3
	NDims2DConvolution = 4
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
	ops.BaseOperator

	autoPad     AutoPadSetting
	dilations   []int
	group       int
	kernelShape []int
	pads        []int
	strides     []int
}

// newConv creates a new conv operator.
func newConv(version int, typeConstraints [][]tensor.DataType) ops.Operator {
	return &Conv{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinConvInputs,
			MaxConvInputs,
			typeConstraints,
			"conv",
		),
		autoPad: NotSet,
	}
}

// Init initializes the conv operator.
func (c *Conv) Init(n *onnx.NodeProto) error {
	var err error

	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "auto_pad":
			c.autoPad = AutoPadSetting(attr.GetS())
		case "dilations":
			c.dilations, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), c)
			}
		case "group":
			c.group = int(attr.GetI())
			if c.group != 1 {
				return ops.ErrUnsupportedAttribute(attr.GetName(), c)
			}
		case "kernel_shape":
			c.kernelShape, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), c)
			}
		case "pads":
			c.pads, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), c)
			}
		case "strides":
			c.strides, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), c)
			}
		default:
			return ops.ErrUnsupportedAttribute(attr.GetName(), c)
		}
	}

	return nil
}

// Apply applies the conv operator.
func (c *Conv) Apply(inputs []*tensor.Tensor) ([]*tensor.Tensor, error) {
	x, kernel, bias := inputs[0], inputs[1], inputs[2]

	if x.Rank() != NDims1DConvolution && x.Rank() != NDims2DConvolution {
		return nil, ops.ErrInvalidInput("the convolution operator currently only supports 1D or 2D convolution, i.e. shape [N x C x H (x W)]", c.BaseOperator)
	}

	if kernel.Rank() != x.Rank() || (kernel.Shape()[1] != x.Shape()[1] && kernel.Shape()[1] != 1) {
		return nil, ops.ErrInvalidInput("the kernel must have shape [M x C x kH (x kW)] matching the input channels", c.BaseOperator)
	}

	if bias != nil && (bias.Rank() != 1 || bias.Shape()[0] != kernel.Shape()[0]) {
		return nil, ops.ErrInvalidInput("the bias must have one value per kernel", c.BaseOperator)
	}

	params, err := c.convParams(x.Shape(), kernel.Shape())
	if err != nil {
		return nil, err
	}

	var out *tensor.Tensor

	switch x.DataType() {
	case tensor.Float32:
		out = convolve[float32](x, kernel, bias, params)
	case tensor.Float64:
		out = convolve[float64](x, kernel, bias, params)
	default:
		return nil, ops.ErrInvalidInputType(0, x.DataType().String(), c.BaseOperator)
	}

	if x.Rank() == NDims1DConvolution {
		shape := out.Shape()

		out, err = out.Reshape(shape[0], shape[1], shape[3])
		if err != nil {
			return nil, err
		}
	}

	return []*tensor.Tensor{out}, nil
}

// convParams holds the convolution settings for a 2D convolution. A 1D convolution is
// computed as a 2D convolution over an image with height 1.
type convParams struct {
	kernel    [2]int
	strides   [2]int
	dilations [2]int
	padsBegin [2]int
	padsEnd   [2]int
}

// convParams resolves the attributes of the operator, and their defaults, for an input
// and kernel with the given shapes.
func (c *Conv) convParams(xShape, kernelShape tensor.Shape) (convParams, error) {
	nSpatialDims := len(xShape) - nNonSpatialDims

	kernel := kernelShape[nNonSpatialDims:]
	strides := orDefault(c.strides, nSpatialDims, 1)
	dilations := orDefault(c.dilations, nSpatialDims, 1)
	pads := orDefault(c.pads, 2*nSpatialDims, 0)

	if len(c.kernelShape) > 0 && !tensor.Shape(c.kernelShape).Equal(kernel) {
		return convParams{}, ops.ErrInvalidInput("kernel_shape does not match the shape of the kernel", c.BaseOperator)
	}

	if len(strides) != nSpatialDims || len(dilations) != nSpatialDims || len(pads) != 2*nSpatialDims {
		return convParams{}, ops.ErrInvalidInput("strides, dilations and pads must match the spatial dimensions", c.BaseOperator)
	}

	params := convParams{
		kernel:    [2]int{1, 1},
		strides:   [2]int{1, 1},
		dilations: [2]int{1, 1},
	}

	// A 1D convolution fills the second (width) dimension, leaving a height of 1.
	offset := 2 - nSpatialDims

	for i := 0; i < nSpatialDims; i++ {
		params.kernel[offset+i] = kernel[i]
		params.strides[offset+i] = strides[i]
		params.dilations[offset+i] = dilations[i]
		params.padsBegin[offset+i] = pads[i]
		params.padsEnd[offset+i] = pads[i+nSpatialDims]

		if c.autoPad == SameUpper || c.autoPad == SameLower {
			inputSize := xShape[nNonSpatialDims+i]
			dilatedKernel := (kernel[i]-1)*dilations[i] + 1
			outputSize := (inputSize + strides[i] - 1) / strides[i]
			padNeeded := max(0, (outputSize-1)*strides[i]+dilatedKernel-inputSize)

			padHead := padNeeded / 2
			if c.autoPad == SameLower {
				padHead = (padNeeded + 1) / 2
			}

			params.padsBegin[offset+i] = padHead
			params.padsEnd[offset+i] = padNeeded - padHead
		} else if c.autoPad == Valid {
			params.padsBegin[offset+i] = 0
			params.padsEnd[offset+i] = 0
		}
	}

	return params, nil
}

func orDefault(values []int, length, defaultValue int) []int {
	if len(values) > 0 {
		return values
	}

	defaults := make([]int, length)
	for i := range defaults {
		defaults[i] = defaultValue
	}

	return defaults
}

// convolve computes the 2D convolution of x [N x C x H x W] with kernel [M x C x kH x kW],
// treating 3D inputs as having height 1. A kernel with a single channel is applied to every
// input channel. Every output element is summed over channels, kernel rows and kernel
// columns in that order; positions in the padding contribute 0.
func convolve[T ops.FloatType](x, kernel, bias *tensor.Tensor, p convParams) *tensor.Tensor {
	xShape, kernelShape := x.Shape(), kernel.Shape()
	nBatches, nChannels := xShape[0], xShape[1]
	nKernels, kernelChannels := kernelShape[0], kernelShape[1]

	height, width := 1, xShape[len(xShape)-1]
	if len(xShape) == NDims2DConvolution {
		height = xShape[2]
	}

	kernelH, kernelW := p.kernel[0], p.kernel[1]
	outH := (height+p.padsBegin[0]+p.padsEnd[0]-((kernelH-1)*p.dilations[0]+1))/p.strides[0] + 1
	outW := (width+p.padsBegin[1]+p.padsEnd[1]-((kernelW-1)*p.dilations[1]+1))/p.strides[1] + 1

	xData, kernelData := tensor.Data[T](x), tensor.Data[T](kernel)
	out := make([]T, nBatches*nKernels*outH*outW)

	var biasData []T
	if bias != nil {
		biasData = tensor.Data[T](bias)
	}

	position := 0

	for b := 0; b < nBatches; b++ {
		for m := 0; m < nKernels; m++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					var sum T

					for ch := 0; ch < nChannels; ch++ {
						image := xData[(b*nChannels+ch)*height*width:]
						weights := kernelData[(m*kernelChannels+ch%kernelChannels)*kernelH*kernelW:]

						for i := 0; i < kernelH; i++ {
							y := oh*p.strides[0] + i*p.dilations[0] - p.padsBegin[0]
							if y < 0 || y >= height {
								continue
							}

							for j := 0; j < kernelW; j++ {
								xPos := ow*p.strides[1] + j*p.dilations[1] - p.padsBegin[1]
								if xPos < 0 || xPos >= width {
									continue
								}

								sum += image[y*width+xPos] * weights[i*kernelW+j]
							}
						}
					}

					if biasData != nil {
						sum += biasData[m]
					}

					out[position] = sum
					position++
				}
			}
		}
	}

	return tensor.New(out, nBatches, nKernels, outH, outW)
}
