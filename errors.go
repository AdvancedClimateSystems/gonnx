package gonnx

import (
	"errors"
	"fmt"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/advancedclimatesystems/gonnx/onnx"
)

var errModel = errors.New("gonnx model error")

// ErrUnsupportedOperator is returned for a model that uses an operator gonnx does not
// implement.
var ErrUnsupportedOperator = ops.ErrUnsupportedOperator

// ErrUnsupportedOpsetVersion is returned for a model with an opset version gonnx does not
// support.
var ErrUnsupportedOpsetVersion = ops.ErrUnsupportedOpsetVersion

type InvalidShapeError struct {
	expected onnx.Shape
	actual   []int
}

func (i InvalidShapeError) Error() string {
	return fmt.Sprintf("invalid shape error expected: %v actual %v", i.expected, i.actual)
}

func ErrInvalidShape(expected onnx.Shape, actual []int) error {
	return InvalidShapeError{
		expected: expected,
		actual:   actual,
	}
}

// ErrModel is used for when an error ocured during setup of running onnx models.
// The user can specify a formatted message using the standard formatting rules.
func ErrModel(format string, a ...any) error {
	return fmt.Errorf("%w: %s", errModel, fmt.Sprintf(format, a...))
}
