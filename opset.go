package gonnx

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/advancedclimatesystems/gonnx/ops/abs"
	"github.com/advancedclimatesystems/gonnx/ops/acos"
	"github.com/advancedclimatesystems/gonnx/ops/acosh"
	"github.com/advancedclimatesystems/gonnx/ops/add"
	"github.com/advancedclimatesystems/gonnx/ops/and"
	"github.com/advancedclimatesystems/gonnx/ops/argmax"
	"github.com/advancedclimatesystems/gonnx/ops/asin"
	"github.com/advancedclimatesystems/gonnx/ops/asinh"
	"github.com/advancedclimatesystems/gonnx/ops/atan"
)

const (
	MinSupportedOpset = 7
	MaxSupportedOpset = 13
)

// OpGetter is a function that gets an operator based on a string.
type OpGetter func(string) (ops.Operator, error)

type OperatorVersions map[int64]func() ops.Operator

var operators = map[string]OperatorVersions{
	"Abs": {
		6:  abs.NewAbs6,
		13: abs.NewAbs13,
	},
	"Acos": {
		7: acos.NewAcos7,
	},
	"Acosh": {
		9: acosh.NewAcosh9,
	},
	"Add": {
		7:  add.NewAdd7,
		13: add.NewAdd13,
	},
	"And": {
		7: and.NewAnd7,
	},
	"ArgMax": {
		11: argmax.NewArgMax11,
		12: argmax.NewArgMax12,
		13: argmax.NewArgMax13,
	},
	"Asin": {
		7: asin.NewAsin7,
	},
	"Asinh": {
		9: asinh.NewAsinh9,
	},
	"Atan": {
		7: atan.NewAtan7,
	},
	"Atanh":           {},
	"Cast":            {},
	"Concat":          {},
	"Constant":        {},
	"ConstantOfShape": {},
	"Conv":            {},
	"Cos":             {},
	"Cosh":            {},
	"Div":             {},
	"Equal":           {},
	"Expand":          {},
	"Flatten":         {},
	"Gather":          {},
	"Gemm":            {},
	"Greater":         {},
	"GreaterOrEqual":  {},
	"GRU":             {},
	"Less":            {},
	"LessOrEqual":     {},
	"LinearRegressor": {},
	"LogSoftmax":      {},
	"LSTM":            {},
	"MatMul":          {},
	"Mul":             {},
	"Not":             {},
	"Or":              {},
	"PRelu":           {},
	"ReduceMax":       {},
	"ReduceMin":       {},
	"Relu":            {},
	"Reshape":         {},
	"RNN":             {},
	"Scaler":          {},
	"Shape":           {},
	"Sigmoid":         {},
	"Sin":             {},
	"Sinh":            {},
	"Slice":           {},
	"Softmax":         {},
	"Squeeze":         {},
	"Sub":             {},
	"Tan":             {},
	"Tanh":            {},
	"Transpose":       {},
	"Unsqueeze":       {},
	"Xor":             {},
}

// ResolveOperatorGetter resolves the getter for operators based on the opset version.
func ResolveOperatorGetter(opsetID int64) (OpGetter, error) {
	if opsetID < MinSupportedOpset || opsetID > MaxSupportedOpset {
		return nil, ops.ErrUnsupportedOpsetVersion
	}

	//TODO: create new OpGetter based on opsetID
	if getOperator, ok := operatorGetters[opsetID]; ok {
		return getOperator, nil
	}

	return nil, ops.ErrUnsupportedOpsetVersion
}
