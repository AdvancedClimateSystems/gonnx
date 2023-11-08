package opset13

import (
	"github.com/advancedclimatesystems/gonnx/ops"
)

var operators13 = map[string]func() ops.Operator{
	"Abs":             newAbs,
	"Add":             newAdd,
	"Cast":            newCast,
	"Concat":          newConcat,
	"Constant":        newConstant,
	"ConstantOfShape": newConstantOfShape,
	"Cos":             newCos,
	"Div":             newDiv,
	"Gather":          newGather,
	"Gemm":            newGemm,
	"GRU":             newGRU,
	"MatMul":          newMatMul,
	"Mul":             newMul,
	"PRelu":           newPRelu,
	"Relu":            newRelu,
	"Reshape":         newReshape,
	"Scaler":          newScaler,
	"Shape":           newShape,
	"Sigmoid":         newSigmoid,
	"Slice":           newSlice,
	"Squeeze":         newSqueeze,
	"Sub":             newSub,
	"Tanh":            newTanh,
	"Transpose":       newTranspose,
	"Unsqueeze":       newUnsqueeze,
}

// GetOperator maps strings as found in the ModelProto to Operators from opset 13.
func GetOperator(operatorType string) (ops.Operator, error) {
	if opInit, ok := operators13[operatorType]; ok {
		return opInit(), nil
	}

	return nil, ops.ErrUnknownOperatorType(operatorType)
}

// GetOpNames returns a list with operator names for opset 13.
func GetOpNames() []string {
	opList := make([]string, 0, len(operators13))

	for opName := range operators13 {
		opList = append(opList, opName)
	}

	return opList
}
