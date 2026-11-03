package gonnx

import (
	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/internal/ops/abs"
	"github.com/advancedclimatesystems/gonnx/internal/ops/acos"
	"github.com/advancedclimatesystems/gonnx/internal/ops/acosh"
	"github.com/advancedclimatesystems/gonnx/internal/ops/add"
	"github.com/advancedclimatesystems/gonnx/internal/ops/and"
	"github.com/advancedclimatesystems/gonnx/internal/ops/argmax"
	"github.com/advancedclimatesystems/gonnx/internal/ops/asin"
	"github.com/advancedclimatesystems/gonnx/internal/ops/asinh"
	"github.com/advancedclimatesystems/gonnx/internal/ops/atan"
	"github.com/advancedclimatesystems/gonnx/internal/ops/atanh"
	"github.com/advancedclimatesystems/gonnx/internal/ops/cast"
	"github.com/advancedclimatesystems/gonnx/internal/ops/concat"
	"github.com/advancedclimatesystems/gonnx/internal/ops/constant"
	"github.com/advancedclimatesystems/gonnx/internal/ops/constantofshape"
	"github.com/advancedclimatesystems/gonnx/internal/ops/conv"
	"github.com/advancedclimatesystems/gonnx/internal/ops/cos"
	"github.com/advancedclimatesystems/gonnx/internal/ops/cosh"
	"github.com/advancedclimatesystems/gonnx/internal/ops/cumsum"
	"github.com/advancedclimatesystems/gonnx/internal/ops/div"
	"github.com/advancedclimatesystems/gonnx/internal/ops/equal"
	"github.com/advancedclimatesystems/gonnx/internal/ops/erf"
	"github.com/advancedclimatesystems/gonnx/internal/ops/expand"
	"github.com/advancedclimatesystems/gonnx/internal/ops/flatten"
	"github.com/advancedclimatesystems/gonnx/internal/ops/gather"
	"github.com/advancedclimatesystems/gonnx/internal/ops/gemm"
	"github.com/advancedclimatesystems/gonnx/internal/ops/greater"
	"github.com/advancedclimatesystems/gonnx/internal/ops/greaterorequal"
	"github.com/advancedclimatesystems/gonnx/internal/ops/gru"
	"github.com/advancedclimatesystems/gonnx/internal/ops/identity"
	"github.com/advancedclimatesystems/gonnx/internal/ops/less"
	"github.com/advancedclimatesystems/gonnx/internal/ops/lessorequal"
	"github.com/advancedclimatesystems/gonnx/internal/ops/linearregressor"
	"github.com/advancedclimatesystems/gonnx/internal/ops/logsoftmax"
	"github.com/advancedclimatesystems/gonnx/internal/ops/lstm"
	"github.com/advancedclimatesystems/gonnx/internal/ops/matmul"
	"github.com/advancedclimatesystems/gonnx/internal/ops/mul"
	"github.com/advancedclimatesystems/gonnx/internal/ops/not"
	"github.com/advancedclimatesystems/gonnx/internal/ops/or"
	"github.com/advancedclimatesystems/gonnx/internal/ops/pow"
	"github.com/advancedclimatesystems/gonnx/internal/ops/prelu"
	"github.com/advancedclimatesystems/gonnx/internal/ops/reducemax"
	"github.com/advancedclimatesystems/gonnx/internal/ops/reducemean"
	"github.com/advancedclimatesystems/gonnx/internal/ops/reducemin"
	"github.com/advancedclimatesystems/gonnx/internal/ops/relu"
	"github.com/advancedclimatesystems/gonnx/internal/ops/reshape"
	"github.com/advancedclimatesystems/gonnx/internal/ops/rnn"
	"github.com/advancedclimatesystems/gonnx/internal/ops/scaler"
	"github.com/advancedclimatesystems/gonnx/internal/ops/shape"
	"github.com/advancedclimatesystems/gonnx/internal/ops/sigmoid"
	"github.com/advancedclimatesystems/gonnx/internal/ops/sin"
	"github.com/advancedclimatesystems/gonnx/internal/ops/sinh"
	"github.com/advancedclimatesystems/gonnx/internal/ops/slice"
	"github.com/advancedclimatesystems/gonnx/internal/ops/softmax"
	"github.com/advancedclimatesystems/gonnx/internal/ops/sqrt"
	"github.com/advancedclimatesystems/gonnx/internal/ops/squeeze"
	"github.com/advancedclimatesystems/gonnx/internal/ops/sub"
	"github.com/advancedclimatesystems/gonnx/internal/ops/tan"
	"github.com/advancedclimatesystems/gonnx/internal/ops/tanh"
	"github.com/advancedclimatesystems/gonnx/internal/ops/transpose"
	"github.com/advancedclimatesystems/gonnx/internal/ops/unsqueeze"
	"github.com/advancedclimatesystems/gonnx/internal/ops/where"
	"github.com/advancedclimatesystems/gonnx/internal/ops/xor"
)

const (
	MinSupportedOpset = 7
	MaxSupportedOpset = 13
)

// opset is a set of operators matching a certain opset version.
type opset map[string]func() ops.Operator

var operators = map[string]ops.OperatorVersions{
	"Abs":             abs.GetVersions(),
	"Acos":            acos.GetVersions(),
	"Acosh":           acosh.GetVersions(),
	"Add":             add.GetVersions(),
	"And":             and.GetVersions(),
	"ArgMax":          argmax.GetVersions(),
	"Asin":            asin.GetVersions(),
	"Asinh":           asinh.GetVersions(),
	"Atan":            atan.GetVersions(),
	"Atanh":           atanh.GetVersions(),
	"Cast":            cast.GetVersions(),
	"Concat":          concat.GetVersions(),
	"Constant":        constant.GetVersions(),
	"ConstantOfShape": constantofshape.GetVersions(),
	"Conv":            conv.GetVersions(),
	"Cos":             cos.GetVersions(),
	"Cosh":            cosh.GetVersions(),
	"CumSum":          cumsum.GetVersions(),
	"Div":             div.GetVersions(),
	"Equal":           equal.GetVersions(),
	"Erf":             erf.GetVersions(),
	"Expand":          expand.GetVersions(),
	"Flatten":         flatten.GetVersions(),
	"Gather":          gather.GetVersions(),
	"Gemm":            gemm.GetVersions(),
	"Greater":         greater.GetVersions(),
	"GreaterOrEqual":  greaterorequal.GetVersions(),
	"GRU":             gru.GetVersions(),
	"Identity":        identity.GetVersions(),
	"Less":            less.GetVersions(),
	"LessOrEqual":     lessorequal.GetVersions(),
	"LinearRegressor": linearregressor.GetVersions(),
	"LogSoftmax":      logsoftmax.GetVersions(),
	"LSTM":            lstm.GetVersions(),
	"MatMul":          matmul.GetVersions(),
	"Mul":             mul.GetVersions(),
	"Not":             not.GetVersions(),
	"Or":              or.GetVersions(),
	"Pow":             pow.GetVersions(),
	"PRelu":           prelu.GetVersions(),
	"ReduceMax":       reducemax.GetVersions(),
	"ReduceMean":      reducemean.GetVersions(),
	"ReduceMin":       reducemin.GetVersions(),
	"Relu":            relu.GetVersions(),
	"Reshape":         reshape.GetVersions(),
	"RNN":             rnn.GetVersions(),
	"Scaler":          scaler.GetVersions(),
	"Shape":           shape.GetVersions(),
	"Sigmoid":         sigmoid.GetVersions(),
	"Sin":             sin.GetVersions(),
	"Sinh":            sinh.GetVersions(),
	"Slice":           slice.GetVersions(),
	"Softmax":         softmax.GetVersions(),
	"Sqrt":            sqrt.GetVersions(),
	"Squeeze":         squeeze.GetVersions(),
	"Sub":             sub.GetVersions(),
	"Tan":             tan.GetVersions(),
	"Tanh":            tanh.GetVersions(),
	"Transpose":       transpose.GetVersions(),
	"Unsqueeze":       unsqueeze.GetVersions(),
	"Xor":             xor.GetVersions(),
	"Where":           where.GetVersions(),
}

// getClosestOperatorVersion resolves, given a certain opset version, the operator version that is closest
// to that version, going downwards. So if the opset version is 13, and an operator has version 13, this
// one is used. If the opset version is 13, and an operator has versions 7 and 14, version 7 is used, as
// it is the closest opset version going downwards.
func getClosestOperatorVersion(opsetID int64, versions ops.OperatorVersions) func() ops.Operator {
	for closestOpset := opsetID; closestOpset >= 1; closestOpset-- {
		if operator, ok := versions[closestOpset]; ok {
			return operator
		}
	}

	return nil
}

// resolveOpset resolves the opset with all closest operator versions for the given opset version.
func resolveOpset(opsetID int64) (opset, error) {
	if opsetID < MinSupportedOpset || opsetID > MaxSupportedOpset {
		return nil, ops.ErrUnsupportedOpsetVersion
	}

	resolved := opset{}

	for operatorName, operatorVersions := range operators {
		operator := getClosestOperatorVersion(opsetID, operatorVersions)
		if operator == nil {
			continue
		}

		resolved[operatorName] = operator
	}

	return resolved, nil
}
