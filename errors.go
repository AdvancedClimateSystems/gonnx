package gonnx

import (
	"errors"
	"fmt"
)

var (
	errInvalidShape    = errors.New("input shape does not match")
	errSetOutputTensor = errors.New("could not set output tensor")
	errModel           = errors.New("gonnx model error")
)

type InvalidShapeError struct {
	expected []int
	actual   []int
}

func (i InvalidShapeError) Error() string {
	return fmt.Sprintf("invalid shape error expected: %v actual %v. mehtod %s", i.expected, i.actual)
}

func ErrInvalidShape(expected, actual []int) error {
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
