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

// TODO weird description.
func ErrInvalidShape(format string, a ...any) error {
	return fmt.Errorf("%w: %s", errInvalidShape, fmt.Sprintf(format, a...))
}

// ErrModel is used for when an error ocured during setup of running onnx models.
// The user can specify a formatted message using the standard formatting rules.
func ErrModel(format string, a ...any) error {
	return fmt.Errorf("%w: %s", errModel, fmt.Sprintf(format, a...))
}
