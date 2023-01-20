package ops

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewSlicer(t *testing.T) {
	tests := []struct {
		start         int
		options       []int
		expectedStart int
		expectedEnd   int
		expectedStep  int
	}{
		{0, []int{}, 0, 1, 1},
		{1, []int{}, 1, 2, 1},
		{0, []int{2}, 0, 2, 1},
		{0, []int{2, 2}, 0, 2, 2},
	}

	for _, test := range tests {
		slicer := NewSlicer(test.start, test.options...)
		assert.Equal(t, test.expectedStart, slicer.Start())
		assert.Equal(t, test.expectedEnd, slicer.End())
		assert.Equal(t, test.expectedStep, slicer.Step())
	}
}
