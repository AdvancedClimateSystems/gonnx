package ops

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/tensor"
	"github.com/stretchr/testify/assert"
)

func TestTanhActivation(t *testing.T) {
	tIn := tensor.New([]float32{1, 2, 3, 4}, 2, 2)
	tOut, err := Tanh(tIn)

	assert.Nil(t, err)
	assert.Equal(t, []float32{0.7615942, 0.9640276, 0.9950548, 0.9993293}, tOut.Data())
}

func TestSigmoidActivation(t *testing.T) {
	tIn := tensor.New([]float32{1, 2, 3, 4}, 2, 2)
	tOut, err := Sigmoid(tIn)

	assert.Nil(t, err)
	assert.InDeltaSlice(t, []float32{0.7310586, 0.880797, 0.95257413, 0.98201376}, tOut.Data(), 1e-6)
}
