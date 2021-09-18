package idx

import (
	"bytes"
	"compress/gzip"
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

// assertEqualsInput is a helper to run assertions verifying that the IDX
// passed in equals the test input specified globally.
func assertEqualsInput(t *testing.T, i *IDX) {
	require.NotNil(t, i)

	require.Len(t, i.Dimensions, 2)
	assert.Equal(t, uint32(3), i.Dimensions[0])
	assert.Equal(t, uint32(4), i.Dimensions[1])
	assert.EqualValues(t, i.Data, input[12:])
}

func TestReadFilePlain(t *testing.T) {
	tmpFile, err := ioutil.TempFile(os.TempDir(), "goml-idx-test-*.idx")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())

	tmpFile.Write(input)

	i, err := ReadFile(tmpFile.Name())
	require.NoError(t, err)
	assertEqualsInput(t, i)
}

func TestReadFileGz(t *testing.T) {
	tmpFile, err := ioutil.TempFile(os.TempDir(), "goml-idx-test-*.idx.gz")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())

	gw := gzip.NewWriter(tmpFile)
	gw.Write(input)
	require.NoError(t, gw.Close())

	i, err := ReadFile(tmpFile.Name())
	require.NoError(t, err)
	assertEqualsInput(t, i)
}

func TestRead(t *testing.T) {
	i, err := Read(bytes.NewReader(input))
	require.NoError(t, err)
	assertEqualsInput(t, i)
}
