package gonnx

// InvalidShapeError is used when the shape of an input tensor does not match the expectation.
const InvalidShapeError = "input shape does not match for %v: expected %v but got %v"

// SetOutputTensorsError is used when the output of an operation could not be set.
const SetOutputTensorsError = "could not set output tensors, expected %v tensors but got %v"
