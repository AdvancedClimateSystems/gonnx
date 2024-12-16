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
	"github.com/advancedclimatesystems/gonnx/ops/less"
	"github.com/advancedclimatesystems/gonnx/ops/lessorequal"
	"github.com/advancedclimatesystems/gonnx/ops/linearregressor"
	"github.com/advancedclimatesystems/gonnx/ops/logsoftmax"
	"github.com/advancedclimatesystems/gonnx/ops/lstm"
	"github.com/advancedclimatesystems/gonnx/ops/matmul"
	"github.com/advancedclimatesystems/gonnx/ops/mul"
	"github.com/advancedclimatesystems/gonnx/ops/not"
	"github.com/advancedclimatesystems/gonnx/ops/or"
	"github.com/advancedclimatesystems/gonnx/ops/prelu"
	"github.com/advancedclimatesystems/gonnx/ops/reducemax"
	"github.com/advancedclimatesystems/gonnx/ops/reducemin"
	"github.com/advancedclimatesystems/gonnx/ops/relu"
	"github.com/advancedclimatesystems/gonnx/ops/reshape"
	"github.com/advancedclimatesystems/gonnx/ops/rnn"
	"github.com/advancedclimatesystems/gonnx/ops/scaler"
	"github.com/advancedclimatesystems/gonnx/ops/shape"
	"github.com/advancedclimatesystems/gonnx/ops/sigmoid"
	"github.com/advancedclimatesystems/gonnx/ops/sin"
	"github.com/advancedclimatesystems/gonnx/ops/sinh"
	"github.com/advancedclimatesystems/gonnx/ops/slice"
	"github.com/advancedclimatesystems/gonnx/ops/softmax"
	"github.com/advancedclimatesystems/gonnx/ops/squeeze"
	"github.com/advancedclimatesystems/gonnx/ops/sub"
	"github.com/advancedclimatesystems/gonnx/ops/tan"
	"github.com/advancedclimatesystems/gonnx/ops/tanh"
	"github.com/advancedclimatesystems/gonnx/ops/transpose"
	"github.com/advancedclimatesystems/gonnx/ops/unsqueeze"
	"github.com/advancedclimatesystems/gonnx/ops/xor"
)

const (
	MinSupportedOpset = 7
	MaxSupportedOpset = 13
)

// Opset is a set of operators matching a certain opset version.
type Opset map[string]func() ops.Operator

var operators = map[string]ops.OperatorVersions{
	"Abs":             abs.GetAbsVersions(),
	"Acos":            acos.GetAcosVersions(),
	"Acosh":           acosh.GetAcoshVersions(),
	"Add":             add.GetAddVersions(),
	"And":             and.GetAndVersions(),
	"ArgMax":          argmax.GetArgMaxVersions(),
	"Asin":            asin.GetAsinVersions(),
	"Asinh":           asinh.GetAsinhVersions(),
	"Atan":            atan.GetAtanVersions(),
	"Atanh":           atanh.GetAtanhVersions(),
	"Cast":            cast.GetCastVersions(),
	"Concat":          concat.GetConcatVersions(),
	"Constant":        constant.GetConstantVersions(),
	"ConstantOfShape": constantofshape.GetConstantOfShapeVersions(),
	"Conv":            conv.GetConvVersions(),
	"Cos":             cos.GetCosVersions(),
	"Cosh":            cosh.GetCoshVersions(),
	"Div":             div.GetDivVersions(),
	"Equal":           equal.GetEqualVersions(),
	"Expand":          expand.GetExpandVersions(),
	"Flatten":         flatten.GetFlattenVersions(),
	"Gather":          gather.GetGatherVersions(),
	"Gemm":            gemm.GetGemmVersions(),
	"Greater":         greater.GetGreaterVersions(),
	"GreaterOrEqual":  greaterorequal.GetGreaterOrEqualVersions(),
	"GRU":             gru.GetGRUVersions(),
	"Less":            less.GetLessVersions(),
	"LessOrEqual":     lessorequal.GetLessOrEqualVersions(),
	"LinearRegressor": linearregressor.GetLinearRegressorVersions(),
	"LogSoftmax":      logsoftmax.GetLogSoftmaxVersions(),
	"LSTM":            lstm.GetLSTMVersions(),
	"MatMul":          matmul.GetMatMulVersions(),
	"Mul":             mul.GetMulVersions(),
	"Not":             not.GetNotVersions(),
	"Or":              or.GetOrVersions(),
	"PRelu":           prelu.GetPReluVersions(),
	"ReduceMax":       reducemax.GetReduceMaxVersions(),
	"ReduceMin":       reducemin.GetReduceMinVersions(),
	"Relu":            relu.GetReluVersions(),
	"Reshape":         reshape.GetReshapeVersions(),
	"RNN":             rnn.GetRNNVersions(),
	"Scaler":          scaler.GetScalerVersions(),
	"Shape":           shape.GetShapeVersions(),
	"Sigmoid":         sigmoid.GetSigmoidVersions(),
	"Sin":             sin.GetSinVersions(),
	"Sinh":            sinh.GetSinhVersions(),
	"Slice":           slice.GetSliceVersions(),
	"Softmax":         softmax.GetSoftmaxVersions(),
	"Squeeze":         squeeze.GetSqueezeVersions(),
	"Sub":             sub.GetSubVersions(),
	"Tan":             tan.GetTanVersions(),
	"Tanh":            tanh.GetTanhVersions(),
	"Transpose":       transpose.GetTransposeVersions(),
	"Unsqueeze":       unsqueeze.GetUnsqueezeVersions(),
	"Xor":             xor.GetXorVersions(),
}

// GetClosestOperatorVersion resolves, given a certain opset version, the operator version that is closest
// to that version, going downwards. So if the opset version is 13, and an operator has version 13, this
// one is used. If the opset version is 13, and an operator has versions 7 and 14, version 7 is used, as
// it is the closest opset version going downwards.
func GetClosestOperatorVersion(opsetID int64, versions ops.OperatorVersions) func() ops.Operator {
	for closestOpset := opsetID; closestOpset >= 1; closestOpset-- {
		if operator, ok := versions[closestOpset]; ok {
			return operator
		}
	}

	return nil
}

// ResolveOpset resolves the opset with all closest operator versions for the given opset version.
func ResolveOpset(opsetID int64) (Opset, error) {
	if opsetID < MinSupportedOpset || opsetID > MaxSupportedOpset {
		return nil, ops.ErrUnsupportedOpsetVersion
	}

	opset := map[string]func() ops.Operator{}

	for operatorName, operatorVersions := range operators {
		operator := GetClosestOperatorVersion(opsetID, operatorVersions)
		if operator == nil {
			continue
		}

		opset[operatorName] = operator
	}

	return opset, nil
}
