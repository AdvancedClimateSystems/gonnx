package ops

import (
	"errors"
	"fmt"
	"reflect"

	"gorgonia.org/tensor"
)

type AttributeErrorKind string

const (
	AttributeErrorCount       AttributeErrorKind = "count"
	AttributeErrorInvalid     AttributeErrorKind = "invalid"
	AttributeErrorUnsupported AttributeErrorKind = "unsupported"
)

type AttributeError struct {
	kind           AttributeErrorKind
	attributeCount int
	expectedCount  int
	attributeName  string
	operator       Operator
}

func (t *AttributeError) Error() string {
	switch t.kind {
	case AttributeErrorCount:
		return fmt.Sprintf("%s attribute error: invalid count %d expected %d", t.operator.String(), t.attributeCount, t.expectedCount)
	case AttributeErrorInvalid:
		return fmt.Sprintf("%s attribute error: invalid attribute %s", t.operator.String(), t.attributeName)
	case AttributeErrorUnsupported:
		return fmt.Sprintf("%s attribute error: unsupported attribute %s", t.operator.String(), t.attributeName)
	default:
		return fmt.Sprintf("%s unknown error attribute error kind %s", t.operator.String(), t.kind)
	}
}

func ErrInvalidAttribute(attributeName string, operator Operator) *AttributeError {
	return &AttributeError{attributeName: attributeName, kind: "invalid", operator: operator}
}

func ErrInvalidAttributeCount(expected, actual int, operator Operator) error {
	return &AttributeError{attributeCount: actual, expectedCount: expected, kind: "count", operator: operator}
}

func ErrUnsupportedAttribute(attributeName string, operator Operator) error {
	return &AttributeError{attributeName: attributeName, kind: "unsupported", operator: operator}
}

type TypeAssertError struct {
	expectedType string
	actualType   any
}

func (t *TypeAssertError) Error() string {
	return fmt.Sprintf("type assert error: expected %v, got %v", t.expectedType, reflect.TypeOf(t.actualType))
}

func ErrTypeAssert(expected string, actual any) error {
	return &TypeAssertError{expectedType: expected, actualType: actual}
}

// InvalidInputCountErrTemplate is used to format an error when an operator got
// the wrong amount of input tensors.
const InvalidInputCountErrTemplate = "%v: expected %d input tensors, got %d"

// InvalidOptionalInputCountErrTemplate is used to format an error when an operator got
// the wrong amount of input tensors when optional inputs are present.
const InvalidOptionalInputCountErrTemplate = "%v: expected %d-%d input tensors, got %d"

// UnsupportedInputErrTemplate is used to format an error when an operator got
// the wrong amount of input tensors when optional inputs are present.
const UnsupportedInputErrTemplate = "unsupported input for %v: %v"

// InvalidInputErrTemplate is used to format an error when an operator got
// an invalid input tensor as input.
const InvalidInputErrTemplate = "invalid input tensor for %v: %v"

type InputErrorKind string

const (
	InputErrorType        InputErrorKind = "type"
	InputErrorCount       InputErrorKind = "count"
	InputErrorUnsupported InputErrorKind = "unsupported"
	InputErrorInvalid     InputErrorKind = "invalid"
)

// AxesNotAllInRangeErrTemplate is used to format an error when not all indices
// are within a given range.
const AxesNotAllInRangeErrTemplate = "all indices entries must be in the range -%d <= x < %d"

// UnsupportedDTypeError is used when the DType of a tensor is not supported.
const UnsupportedDtypeErrTemplate = "dtype %v is not supported for operator %v"

type InputError struct {
	kind     InputErrorKind
	operator Operator
	reason   string

	// Attributes for input type error.
	inputNumber int
	actualType  string

	// Attributes for input count error.
	hasOptionalInputs bool
	actualCount       int

	// Attributes for unsupported input error.
	inputName string
}

func (i *InputError) Error() string {
	switch i.kind {
	case InputErrorType:
		return fmt.Sprintf("input %d for op %v does not allow dtype %v", i.inputNumber, i.operator, i.actualType)
	case InputErrorCount:
		if i.hasOptionalInputs {
			return fmt.Sprintf(InvalidOptionalInputCountErrTemplate, i.operator, i.operator.GetMinInputs(), i.operator.GetMaxInputs(), i.actualCount)
		}

		return fmt.Sprintf(InvalidInputCountErrTemplate, i.operator, i.operator.GetMinInputs(), i.actualCount)
	case InputErrorUnsupported:
		return fmt.Sprintf(UnsupportedInputErrTemplate, i.operator, i.inputName)
	case InputErrorInvalid:
		return fmt.Sprintf(InvalidInputErrTemplate, i.operator, i.reason)
	default:
		return fmt.Sprintf("%s unknown error input error kind %s", i.operator.String(), i.kind)
	}
}

func ErrInvalidInputType(inputNumber int, dType string, operator Operator) error {
	return &InputError{
		kind:        InputErrorType,
		operator:    operator,
		inputNumber: inputNumber,
		actualType:  dType,
	}
}

func ErrInvalidInputCount(actual int, operator Operator) error {
	return &InputError{
		kind:        InputErrorCount,
		actualCount: actual,
		operator:    operator,
	}
}

func ErrInvalidOptionalInputCount(actual int, operator Operator) error {
	return &InputError{
		kind:              InputErrorCount,
		hasOptionalInputs: true,
		actualCount:       actual,
		operator:          operator,
	}
}

func ErrUnsupportedInput(inputName string, operator Operator) error {
	return &InputError{
		kind:      InputErrorUnsupported,
		inputName: inputName,
		operator:  operator,
	}
}

func ErrInvalidInput(reason string, operator Operator) error {
	return &InputError{
		kind:     InputErrorInvalid,
		reason:   reason,
		operator: operator,
	}
}

type BroadcastError struct {
	broadcastType string
	shapeA        tensor.Shape
	shapeB        tensor.Shape
	err           error
}

func (b *BroadcastError) Error() string {
	return fmt.Sprintf("%v: could not perform %v, inputs with shape %d and %d.", b.err, b.broadcastType, b.shapeA, b.shapeB)
}

func ErrMultidirBroadcast(shapeA, shapeB tensor.Shape, err error) error {
	return &BroadcastError{
		broadcastType: "multidirectional broadcast",
		shapeA:        shapeA,
		shapeB:        shapeB,
		err:           err,
	}
}

func ErrUnidirBroadcast(shapeA, shapeB tensor.Shape) error {
	return &BroadcastError{
		broadcastType: "Unidirectional broadcast",
		shapeA:        shapeA,
		shapeB:        shapeB,
	}
}

type InvalidTensorError struct {
	reason   string
	operator Operator
}

func (i *InvalidTensorError) Error() string {
	return fmt.Sprintf("%v invalid tensor found, reason: %s", i.operator.String(), i.reason)
}

func ErrInvalidTensor(reason string, operator Operator) error {
	return &InvalidTensorError{reason: reason, operator: operator}
}

var ErrUnsupportedOperator = errors.New("unsupported operator")

func ErrUnknownOperatorType(operatorType string) error {
	return fmt.Errorf("%w: %s", ErrUnsupportedOperator, operatorType)
}

var ErrAxisNotInRange = errors.New("axis out of range")

func ErrNotAllAxesInRange(min, max int) error {
	return fmt.Errorf("%w: all indices entries must be in the range -%d <= x < %d", ErrAxisNotInRange, min, max)
}

func ErrAxisOutOfRange(min, max, actual int) error {
	return fmt.Errorf("%w: axis argument must be in the range -%d <= x < %d, was %d", ErrAxisNotInRange, min, max, actual)
}

var ErrUnsupportedOpsetVersion = errors.New("unsupported opset version")

type DimensionErrorKind string

const (
	DimensionErrorIncompatible DimensionErrorKind = "incompatible"
)

type DimensionError struct {
	kind   DimensionErrorKind
	reason string
}

func (d *DimensionError) Error() string {
	switch d.kind {
	case DimensionErrorIncompatible:
		return fmt.Sprintf("dimensions error: incompatible dimensions")
	default:
		return fmt.Sprintf("dimension error: %s", d.reason)
	}
}

func ErrIncompatibleDimensions() error {
	return &DimensionError{kind: DimensionErrorIncompatible, reason: ""}
}

func ErrDimension(reason string) error {
	return &DimensionError{reason: reason}
}

var (
	ErrCast         = errors.New("cast error")
	ErrInvalidShape = errors.New("invalid shape error")
)

var ErrConversion = errors.New("unable to convert")

func ErrConversionInvalidType(dType tensor.Dtype, newType int32) error {
	return fmt.Errorf("%w: type %v, to %v is invalid", ErrConversion, dType, newType)
}

func ErrConversionNotSupported(dType int32) error {
	return fmt.Errorf("%w: to %v is not supported yet", ErrConversion, dType)
}
