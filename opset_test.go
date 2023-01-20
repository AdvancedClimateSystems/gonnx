package gonnx

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestResolveOperatorGetterFail(t *testing.T) {
	opGetter, err := ResolveOperatorGetter(12)
	assert.Nil(t, opGetter)
	assert.Equal(t, fmt.Errorf("expected opset to be in [13], got operator set 12"), err)
}
