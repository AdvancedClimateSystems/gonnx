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
	"github.com/advancedclimatesystems/gonnx/ops/gather"
	"github.com/advancedclimatesystems/gonnx/ops/gemm"
	"github.com/advancedclimatesystems/gonnx/ops/greater"
	"github.com/advancedclimatesystems/gonnx/ops/greaterorequal"
	"github.com/advancedclimatesystems/gonnx/ops/gru"
)

const (
	MinSupportedOpset = 7
	MaxSupportedOpset = 13
)

// OpGetter is a function that gets an operator based on a string.
type OpGetter func(string) (ops.Operator, error)

var operators = map[string]ops.OperatorVersions{
	"Abs":             abs.AbsVersions,
	"Acos":            acos.AcosVersions,
	"Acosh":           acosh.AcoshVersions,
	"Add":             add.AddVersions,
	"And":             and.AndVersions,
	"ArgMax":          argmax.ArgMaxVersions,
	"Asin":            asin.AsinVersions,
	"Asinh":           asinh.AsinhVersions,
	"Atan":            atan.AtanVersions,
	"Atanh":           atanh.AtanhVersions,
	"Cast":            cast.CastVersions,
	"Concat":          concat.ConcatVersions,
	"Constant":        constant.ConstantVersions,
	"ConstantOfShape": constantofshape.ConstantOfShapeVersions,
	"Conv":            conv.ConvVersions,
	"Cos":             cos.CosVersions,
	"Cosh":            cosh.CoshVersions,
	"Div":             div.DivVersions,
	"Equal":           equal.EqualVersions,
	"Expand":          expand.ExpandVersions,
	"Flatten":         flatten.FlattenVersions,
	"Gather":          gather.GatherVersions,
	"Gemm":            gemm.GemmVersions,
	"Greater":         greater.GreaterVersions,
	"GreaterOrEqual":  greaterorequal.GreaterOrEqualVersions,
	"GRU":             gru.GRUVersions,
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
