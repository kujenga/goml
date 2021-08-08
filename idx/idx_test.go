package idx

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRead(t *testing.T) {
	// Example input manually constructed for testing purposes, based on
	// the specification for IDX files.
	var input = []byte{
		// Magic number: uint8, 2dim
		0, 0, 0x08, 2,
		// 2dmin of lengths, big endian
		0, 0, 0, 3,
		0, 0, 0, 4,
		// 3x4 values
		1, 2, 3,
		4, 5, 6,
		10, 20, 30,
		40, 50, 60,
	}

	i, err := Read(bytes.NewReader(input))
	require.NoError(t, err)
	require.NotNil(t, i)

	require.Len(t, i.Dimensions, 2)
	assert.Equal(t, uint32(3), i.Dimensions[0])
	assert.Equal(t, uint32(4), i.Dimensions[1])
	assert.EqualValues(t, i.Data, input[12:])
}
