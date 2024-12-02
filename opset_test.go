package gonnx

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
)

func TestResolveOpset(t *testing.T) {
	_, err := ResolveOpset(13)
	assert.Nil(t, err)
}

func TestResolveOpsetNotSupported(t *testing.T) {
	opset, err := ResolveOpset(6)
	assert.Nil(t, opset)
	assert.Equal(t, ops.ErrUnsupportedOpsetVersion, err)
}
