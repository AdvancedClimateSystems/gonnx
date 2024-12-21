package pow

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestPowInit(t *testing.T) {
	p := &Pow{}
	err := p.Init(nil)
	assert.Nil(t, err)
}

func TestPow(t *testing.T) {
	tests := []struct {
		version  int64
		backing0 any
		backing1 any
		shapes   [][]int
		expected any
	}{
		{
			13,
			[]float32{0, 1, 2, 3},
			[]float32{1, 1, 1, 1},
			[][]int{{2, 2}, {2, 2}},
			[]float32{0, 1, 2, 3},
		},
		{
			13,
			[]float32{0, 1, 2, 3, 4, 5},
			[]float32{2, 2, 2, 2, 2, 2},
			[][]int{{3, 2}, {3, 2}},
			[]float32{0, 1, 4, 9, 16, 25},
		},
		{
			13,
			[]float32{0, 1},
			[]float32{0, 1, 2, 3},
			[][]int{{2}, {2, 2}},
			[]float32{1, 1, 0, 1},
		},
		{
			13,
			[]int32{1, 2, 3},
			[]int32{4, 5, 6},
			[][]int{{3}, {3}},
			[]int32{1, 32, 729},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing0, test.shapes[0]...),
			ops.TensorWithBackingFixture(test.backing1, test.shapes[1]...),
		}

		pow := powVersions[test.version]()

		res, err := pow.Apply(inputs)
		assert.Nil(t, err)

		assert.Equal(t, test.expected, res[0].Data())
	}
}
