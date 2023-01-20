package ops

import "gorgonia.org/tensor"

// Slicer implements the tensor.Slice interface. It is able to slice the dimension of a tensor.
type Slicer struct {
	start int
	end   int
	step  int
}

// NewSlicer creates a new Slicer object. By default, end will be set to start + 1 and step
// will be set to 1. If options are given, it is assumed that the first element will be the value
// for the end index and the second element the value for the step of the Slicer.
func NewSlicer(start int, options ...int) tensor.Slice {
	end := start + 1
	step := 1

	if len(options) >= 1 {
		end = options[0]
	}

	if len(options) >= 2 {
		step = options[1]
	}

	return &Slicer{
		start: start,
		end:   end,
		step:  step,
	}
}

// Start returns the start of the slice.
func (s *Slicer) Start() int {
	return s.start
}

// End returns the start of the slice.
func (s *Slicer) End() int {
	return s.end
}

// Step returns the step of the slice.
func (s *Slicer) Step() int {
	return s.step
}
