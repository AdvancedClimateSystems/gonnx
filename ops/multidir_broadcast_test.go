package ops

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestMultidirectionalBroadcast(t *testing.T) {
	tests := []struct {
		shapes        [][]int
		expectedShape tensor.Shape
		err           error
	}{
		{
			[][]int{{2}, {2, 2}},
			[]int{2, 2},
			nil,
		},
		{
			[][]int{{2, 3, 4, 5}, {}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{2, 3, 4, 5}, {5}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{4, 5}, {2, 3, 4, 5}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{1, 4, 5}, {2, 3, 1, 1}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{3, 4, 5}, {2, 1, 1, 1}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{1, 4, 5}, {2, 1, 1, 3}},
			nil,
			fmt.Errorf(
				MultidirBroadcastErrTemplate,
				[]int{1, 4, 5},
				[]int{2, 1, 1, 3},
				"incompatible dimensions",
			),
		},
		{
			[][]int{{5}, {2, 3, 4}},
			nil,
			fmt.Errorf(
				MultidirBroadcastErrTemplate,
				[]int{5},
				[]int{2, 3, 4},
				"incompatible dimensions",
			),
		},
	}

	for _, test := range tests {
		A := Float32TensorFixture(test.shapes[0]...)
		B := Float32TensorFixture(test.shapes[1]...)

		newA, newB, err := MultidirectionalBroadcast(A, B)

		assert.Equal(t, test.err, err)

		if err == nil {
			assert.Equal(t, test.expectedShape, newA.Shape())
			assert.Equal(t, test.expectedShape, newB.Shape())
		} else {
			assert.Nil(t, newA)
			assert.Nil(t, newB)
		}
	}
}
