package ops

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestValidateInputs(t *testing.T) {
	tests := []struct {
		op          Operator
		inputs      []tensor.Tensor
		expectedNil int
		err         error
	}{
		{
			&MockOp{
				minInputs:            1,
				maxInputs:            1,
				inputTypeConstraints: [][]tensor.Dtype{{tensor.Float32}},
			},
			PaddedInputsFixture(1, 0),
			0,
			nil,
		},
		{
			&MockOp{
				minInputs:            2,
				maxInputs:            2,
				inputTypeConstraints: [][]tensor.Dtype{{tensor.Float32}, {tensor.Float32}},
			},
			PaddedInputsFixture(2, 0),
			0,
			nil,
		},
		{
			&MockOp{
				minInputs: 3,
				maxInputs: 5,
				inputTypeConstraints: [][]tensor.Dtype{
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
				},
			},
			PaddedInputsFixture(3, 0),
			2,
			nil,
		},
		{
			&MockOp{
				minInputs: 3,
				maxInputs: 5,
				inputTypeConstraints: [][]tensor.Dtype{
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
				},
			},
			PaddedInputsFixture(4, 0),
			1,
			nil,
		},
		{
			&MockOp{
				minInputs:            2,
				maxInputs:            2,
				inputTypeConstraints: [][]tensor.Dtype{{tensor.Float32}, {tensor.Float32}},
			},
			PaddedInputsFixture(1, 0),
			0,
			ErrInvalidInputCount(2, &MockOp{}),
		},
		{
			&MockOp{
				minInputs: 3,
				maxInputs: 5,
				inputTypeConstraints: [][]tensor.Dtype{
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
					{tensor.Float32},
				},
			},
			PaddedInputsFixture(7, 0),
			0,
			ErrInvalidOptionalInputCount(7, &MockOp{}),
		},
		{
			&MockOp{
				minInputs:            2,
				maxInputs:            2,
				inputTypeConstraints: [][]tensor.Dtype{{tensor.Float32}, {tensor.Float64}},
			},
			PaddedInputsFixture(2, 0),
			0,
			ErrInvalidInputType(1, "float32", &MockOp{}),
		},
	}

	for _, test := range tests {
		inputs, err := ValidateInputs(test.op, test.inputs)

		expectedLength := len(test.inputs) + test.expectedNil
		assert.Equal(t, test.err, err)
		assert.Equal(t, expectedLength, len(inputs))

		// Check if the added nodes are all nil.
		for i := len(test.inputs); i < expectedLength; i++ {
			assert.Nil(t, inputs[i])
		}
	}
}

func TestPadInputs(t *testing.T) {
	nodes := PaddedInputsFixture(2, 0)

	paddedNodes := padInputs(nodes, 5)

	assert.Equal(t, len(paddedNodes), 5)
	assert.Nil(t, paddedNodes[2])
	assert.Nil(t, paddedNodes[3])
	assert.Nil(t, paddedNodes[4])
}

// PaddedInputsFixture returns an array of tensor.Tensor with the first nTensors tensors as empty
// tensors and the last nNil tensors with nil elements.
func PaddedInputsFixture(nTensors, nNil int) []tensor.Tensor {
	result := make([]tensor.Tensor, nTensors+nNil)
	i := 0

	for ; i < nTensors; i++ {
		result[i] = tensor.New(tensor.WithBacking([]float32{0.0}))
	}

	for ; i < nTensors+nNil; i++ {
		result[i] = nil
	}

	return result
}

type MockOp struct {
	minInputs            int
	maxInputs            int
	inputTypeConstraints [][]tensor.Dtype
}

func (m *MockOp) Init(_ []*onnx.AttributeProto) error {
	return nil
}

func (m *MockOp) Apply(_ []tensor.Tensor) ([]tensor.Tensor, error) {
	return nil, nil
}

func (m *MockOp) String() string {
	return "mock op"
}

func (m *MockOp) GetMinInputs() int {
	return m.minInputs
}

func (m *MockOp) GetMaxInputs() int {
	return m.maxInputs
}

func (m *MockOp) GetInputTypeConstraints() [][]tensor.Dtype {
	return m.inputTypeConstraints
}

func (m *MockOp) ValidateInputs(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	return inputs, nil
}
