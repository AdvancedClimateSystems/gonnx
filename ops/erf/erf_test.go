package erf

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestErfInit(t *testing.T) {
	e := &Erf{}
	err := e.Init(nil)
	assert.Nil(t, err)
}

func TestErf(t *testing.T) {
	tests := []struct {
		version  int64
		backing  []float32
		shape    []int
		expected []float32
	}{
		{
			9,
			[]float32{-1, -1, 0, 1},
			[]int{2, 2},
			[]float32{-0.8427008, -0.8427008, 0, 0.8427008},
		},
		{
			13,
			[]float32{1, 0.5, 0.0, -0.5},
			[]int{1, 4},
			[]float32{0.8427008, 0.5204999, 0, -0.5204999},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			ops.TensorWithBackingFixture(test.backing, test.shape...),
		}

		erf := erfVersions[test.version]()

		res, err := erf.Apply(inputs)
		assert.Nil(t, err)

		assert.Nil(t, err)
		assert.Equal(t, test.expected, res[0].Data())
	}
}
