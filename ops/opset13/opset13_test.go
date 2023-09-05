package opset13

import (
	"fmt"
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
)

func TestGetOperator(t *testing.T) {
	tests := []struct {
		opType   string
		expected ops.Operator
		err      error
	}{
		{
			"Abs",
			newAbs(),
			nil,
		},
		{
			"Add",
			newAdd(),
			nil,
		},
		{
			"Asin",
			newAsin(),
			nil,
		},
		{
			"Cast",
			newCast(),
			nil,
		},
		{
			"Concat",
			newConcat(),
			nil,
		},
		{
			"Constant",
			newConstant(),
			nil,
		},
		{
			"ConstantOfShape",
			newConstantOfShape(),
			nil,
		},
		{
			"Div",
			newDiv(),
			nil,
		},
		{
			"Gather",
			newGather(),
			nil,
		},
		{
			"Gemm",
			newGemm(),
			nil,
		},
		{
			"GRU",
			newGRU(),
			nil,
		},
		{
			"MatMul",
			newMatMul(),
			nil,
		},
		{
			"Mul",
			newMul(),
			nil,
		},
		{
			"Relu",
			newRelu(),
			nil,
		},
		{
			"Reshape",
			newReshape(),
			nil,
		},
		{
			"Scaler",
			newScaler(),
			nil,
		},
		{
			"Shape",
			newShape(),
			nil,
		},
		{
			"Sigmoid",
			newSigmoid(),
			nil,
		},
		{
			"Slice",
			newSlice(),
			nil,
		},
		{
			"Squeeze",
			newSqueeze(),
			nil,
		},
		{
			"Sub",
			newSub(),
			nil,
		},
		{
			"Tanh",
			newTanh(),
			nil,
		},
		{
			"Transpose",
			newTranspose(),
			nil,
		},
		{
			"Unsqueeze",
			newUnsqueeze(),
			nil,
		},
		{
			"NotYetImplemented",
			nil,
			fmt.Errorf(ops.UnknowOpTypeErrTemplate, "NotYetImplemented"),
		},
	}

	for _, test := range tests {
		op, err := GetOperator(test.opType)

		assert.Equal(t, test.expected, op)
		assert.Equal(t, test.err, err)
	}
}
