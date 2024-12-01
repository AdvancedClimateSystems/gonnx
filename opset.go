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
	"github.com/advancedclimatesystems/gonnx/ops/atanh"
	"github.com/advancedclimatesystems/gonnx/ops/cast"
	"github.com/advancedclimatesystems/gonnx/ops/concat"
	"github.com/advancedclimatesystems/gonnx/ops/constant"
	"github.com/advancedclimatesystems/gonnx/ops/constantofshape"
	"github.com/advancedclimatesystems/gonnx/ops/conv"
	"github.com/advancedclimatesystems/gonnx/ops/cos"
	"github.com/advancedclimatesystems/gonnx/ops/cosh"
	"github.com/advancedclimatesystems/gonnx/ops/div"
	"github.com/advancedclimatesystems/gonnx/ops/equal"
	"github.com/advancedclimatesystems/gonnx/ops/expand"
	"github.com/advancedclimatesystems/gonnx/ops/flatten"
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
		6:  abs.NewAbs6, // Same, but bfloat16 type is added
		13: abs.NewAbs13,
	},
	"Acos": {
		7: acos.NewAcos7,
	},
	"Acosh": {
		9: acosh.NewAcosh9,
	},
	"Add": {
		7:  add.NewAdd7, // Same, but bfloat16 type is added
		13: add.NewAdd13,
	},
	"And": {
		7: and.NewAnd7,
	},
	"ArgMax": {
		11: argmax.NewArgMax11, // Same, but one attribute is added (which we don't support it anyway)
		12: argmax.NewArgMax12, // Same, but bfloat16 type differs
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
	"Atanh": {
		9: atanh.NewAtanh9,
	},
	"Cast": {
		6:  cast.NewCast6, // Same, but string type is added
		9:  cast.NewCast9, // Same, but bfloat16 type differs
		13: cast.NewCast13,
	},
	"Concat": {
		4:  concat.NewConcat4,
		11: concat.NewConcat11, // Same, but bfloat16 type differs
		13: concat.NewConcat13,
	},
	"Constant": {
		1:  constant.NewConstant1,
		9:  constant.NewConstant9,
		11: constant.NewConstant11,
		12: constant.NewConstant12, // Same, but bfloat16 type differs
		13: constant.NewConstant13,
	},
	"ConstantOfShape": {
		9: constantofshape.NewConstantOfShape9,
	},
	"Conv": {
		1:  conv.NewConv1, // Same, but only float16 type differs
		11: conv.NewConv11,
	},
	"Cos": {
		7: cos.NewCos7,
	},
	"Cosh": {
		9: cosh.NewCosh9,
	},
	"Div": {
		7:  div.NewDiv7, // Same, but float16 type differs
		13: div.NewDiv13,
	},
	"Equal": {
		7:  equal.NewEqual7,
		11: equal.NewEqual11, // Same, but float16 type differs
		13: equal.NewEqual13,
	},
	"Expand": {
		8:  expand.NewExpand8, // Same, but float16 type differs
		13: expand.NewExpand13,
	},
	"Flatten": {
		1:  flatten.NewFlatten1,  // Same, but only float types
		9:  flatten.NewFlatten9,  // Same, but negative axis added
		11: flatten.NewFlatten11, // Same, but float16 type differs
		13: flatten.NewFlatten13,
	},
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
