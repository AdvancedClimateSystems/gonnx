package gonnx

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
)

func TestResolveOperatorGetterFail(t *testing.T) {
	opGetter, err := ResolveOperatorGetter(12)
	assert.Nil(t, opGetter)
	assert.Equal(t, ops.ErrUnsupportedOpsetVersion, err)
}
