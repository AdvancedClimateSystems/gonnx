package ops

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestConvertTensorDtype(t *testing.T) {
	tests := []struct {
		tensorIn  tensor.Tensor
		tensorOut tensor.Tensor
		newType   int32
		err       error
	}{
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{1.0, 2.0})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1.0, 2.0})),
			11,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1.0, 2.0})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{1.0, 2.0})),
			1,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int8{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]uint16{1, 2})),
			4,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int16{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]uint8{1, 2})),
			2,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int32{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]uint64{1, 2})),
			13,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int64{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int8{1, 2})),
			3,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]uint8{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int16{1, 2})),
			5,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]uint16{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int32{1, 2})),
			6,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]uint32{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{1.0, 2.0})),
			1,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]uint64{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]uint32{1, 2})),
			12,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{1.0, 2.0})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int64{1, 2})),
			7,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]int64{1, 2})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{1.0, 2.0})),
			1,
			nil,
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]bool{true, false})),
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{1.0, 2.0})),
			1,
			fmt.Errorf("unable to convert tensor of type bool to type 1"),
		},
		{
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{1.0, 2.1})),
			nil,
			8,
			fmt.Errorf("converting to onnx datatype 8 is not supported yet"),
		},
	}

	for _, test := range tests {
		out, err := ConvertTensorDtype(test.tensorIn, test.newType)

		assert.Equal(t, test.err, err)
		if test.err != nil {
			continue
		}

		assert.Equal(t, test.tensorOut, out)
	}
}

func TestCreateNewBacking(t *testing.T) {
	assert.InDeltaSlice(t, []float64{0.5, 0.8}, createNewBacking[float32, float64]([]float32{0.5, 0.8}), 0.00001)
	assert.Equal(t, []int32{1, 2}, createNewBacking[float32, int32]([]float32{1.2, 2.5}))
	assert.Equal(t, []float32{1.0, 2.0}, createNewBacking[int64, float32]([]int64{1, 2}))
}
