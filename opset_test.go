package gonnx

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/internal/ops"
	"github.com/stretchr/testify/assert"
)

func TestResolveOpset(t *testing.T) {
	_, err := resolveOpset(13)
	assert.Nil(t, err)
}

func TestResolveOpsetNotSupported(t *testing.T) {
	resolved, err := resolveOpset(6)
	assert.Nil(t, resolved)
	assert.Equal(t, ops.ErrUnsupportedOpsetVersion, err)
}
