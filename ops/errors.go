package ops

import (
	"errors"
	"fmt"
	"reflect"

	"gorgonia.org/tensor"
)

type AttributeError struct {
	kind           string
	attributeCount int
	expectedCount  int
	attributeName  string
	operator       Operator
}

func (t *AttributeError) Error() string {
	switch t.kind {
	case "count":
		return fmt.Sprintf("operator %s attribute error: invalid count %d expected %d", t.operator.String(), t.attributeCount, t.expectedCount)
	case "name":
		return fmt.Sprintf("operator %s attribute error: invalid attribute %s", t.operator.String(), t.attributeName)
	}

	return fmt.Sprintf("attribute error")
}

func ErrInvalidAttribute(attributeName string, operator Operator) error {
	return &AttributeError{attributeName: attributeName, kind: "count", operator: operator}
}

func ErrInvalidAttributeCount(expected, actual int, operator Operator) error {
	return &AttributeError{attributeCount: actual, expectedCount: expected, kind: "count", operator: operator}
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

// UnknownAttributeErrTemplate is used to format an error
// when an operator finds an unknown attribute during its initialization.
const UnknownAttributeErrTemplate = "%v: unknown attribute: %v"

// UnsupportedAttrErrTemplate is used to format an error when an operator receives
// an attribute that is not supported yet.
const UnsupportedAttrErrTemplate = "%v: %v attribute not supported yet"

// InvalidAttrCountErrTemplate is used to format an error when an operator
// got the wrong amount of attributes.
const InvalidAttrCountErrTemplate = "%v: expected %v attributes, got %d"

// InvalidInputCountErrTemplate is used to format an error when an operator got
// the wrong amount of input tensors.
const InvalidInputCountErrTemplate = "%v: expected %d input tensors, got %d"

// InvalidOptionalInputCountErrTemplate is used to format an error when an operator got
// the wrong amount of input tensors when optional inputs are present.
const InvalidOptionalInputCountErrTemplate = "%v: expected %d-%d input tensors, got %d"

type InvalidInputTypeError struct {
	inputNumber int
	actualType  string
	operator    Operator
}

func (i *InvalidInputTypeError) Error() string {
	return fmt.Sprintf("input %d for op %v does not allow dtype %v", i.inputNumber, i.operator, i.actualType)
}

func ErrInvalidInputType(operator Operator, inputNumber int, dType string) error {
	return &InvalidInputTypeError{
		operator:    operator,
		inputNumber: inputNumber,
		actualType:  dType,
	}
}

type InvalidInputCountError struct {
	hasOptionalInputs bool
	actualCount       int
	operator          Operator
}

func (i *InvalidInputCountError) Error() string {
	if i.hasOptionalInputs {
		return fmt.Sprintf(InvalidOptionalInputCountErrTemplate, i.operator, i.operator.GetMinInputs(), i.operator.GetMaxInputs(), i.actualCount)
	}

	return fmt.Sprintf(InvalidInputCountErrTemplate, i.operator, i.operator.GetMinInputs(), i.actualCount)
}

func ErrInvalidInputCount(operator Operator, actual int) error {
	return &InvalidInputCountError{
		actualCount: actual,
		operator:    operator,
	}
}

func ErrInvalidOptionalInputCount(operator Operator, actual int) error {
	return &InvalidInputCountError{
		hasOptionalInputs: true,
		actualCount:       actual,
		operator:          operator,
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

var ErrIncompatibleDimension = errors.New("incompatible dimensions")

// UnknowOpTypeErrTemplate is used to format an error when the operator type is unknown.
const UnknowOpTypeErrTemplate = "unknown operator type: %v"

// MultidirBroadcastErrTemplate is used to format an error when two inputs cannot be
// broadcasted together with Multidirectional broadcasting.
const MultidirBroadcastErrTemplate = "could not multidir broadcast inputs with shape %d and %d: %v"

// UnidirBroadcastErrTemplate is used to format an error when two inputs cannot be
// broadcasted together with Unidirectional broadcasting.
const UnidirBroadcastErrTemplate = "could not unidir broadcast inputs with shape %d and %d"

// AxisOutOfRangeErrTemplate is used to format an error when an given axis is out of range
// given a certain rank.
const AxisOutOfRangeErrTemplate = "axis argument must be in the range -%d <= x < %d, was %d"

// AxesNotAllInRangeErrTemplate is used to format an error when not all indices
// are within a given range.
const AxesNotAllInRangeErrTemplate = "all indices entries must be in the range -%d <= x < %d"

var ErrAxisNotInRange = errors.New("axis out of range")

func ErrNotAllAxisInRange(min, max int) error {
	return fmt.Errorf("%w: all indices entries must be in the range -%d <= x < %d", ErrAxisNotInRange, min, max)
}

func ErrAxisOutOfRange(min, max, actual int) error {
	return fmt.Errorf("%w: axis argument must be in the range -%d <= x < %d, was %d", ErrAxisNotInRange, min, max, actual)
}
