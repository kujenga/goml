package idx

import (
	"bytes"
	"compress/gzip"
	"fmt"
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

	_, err = tmpFile.Write(input)
	require.NoError(t, err)

	i, err := ReadFile(tmpFile.Name())
	require.NoError(t, err)
	assertEqualsInput(t, i)
}

func TestReadFileMissing(t *testing.T) {
	tmpFile, err := ioutil.TempFile(os.TempDir(), "goml-idx-test-*.idx")
	require.NoError(t, err)
	// Remove the file, so that it is missing.
	require.NoError(t, os.Remove(tmpFile.Name()))

	i, err := ReadFile(tmpFile.Name())
	require.Error(t, err)
	assert.Nil(t, i)
}

func TestReadFileGz(t *testing.T) {
	tmpFile, err := ioutil.TempFile(os.TempDir(), "goml-idx-test-*.idx.gz")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())

	gw := gzip.NewWriter(tmpFile)
	_, err = gw.Write(input)
	require.NoError(t, err)
	require.NoError(t, gw.Close())

	i, err := ReadFile(tmpFile.Name())
	require.NoError(t, err)
	assertEqualsInput(t, i)
}

// TestReadFileGzInvalid tests that the parser does error in an expected manner
// when non-gzip data with a .gz filename is passed in.
func TestReadFileGzInvalid(t *testing.T) {
	tmpFile, err := ioutil.TempFile(os.TempDir(), "goml-idx-test-*.idx.gz")
	require.NoError(t, err)
	defer os.Remove(tmpFile.Name())

	_, err = tmpFile.Write(input)
	require.NoError(t, err)

	i, err := ReadFile(tmpFile.Name())
	require.Error(t, err)
	assert.Nil(t, i)
}

func TestRead(t *testing.T) {
	i, err := Read(bytes.NewReader(input))
	require.NoError(t, err)
	assertEqualsInput(t, i)
}

// TestReadInvalid tests that the parser properly errors in various cases where
// it is handed invalid input.
func TestReadInvalid(t *testing.T) {
	for idx, tc := range []struct {
		input         []byte
		errorContains string
	}{
		{
			input:         nil,
			errorContains: "EOF",
		},
		{
			input:         []byte{},
			errorContains: "EOF",
		},
		{
			input:         []byte{0, 0},
			errorContains: "expected 4",
		},
		{
			input:         []byte{1, 2, 3, 4},
			errorContains: "should be zero",
		},
		{
			input:         []byte{0, 0, 0x16, 2},
			errorContains: "only uint8 data type supported",
		},
		{
			input: []byte{
				// Magic number: uint8, 1dim
				0, 0, 0x08, 3,
				// 2dmin of lengths (mismatched)
				0, 0, 0, 3,
				0, 0, 0, 4,
			},
			errorContains: "EOF",
		},
		{
			input: []byte{
				// Magic number: uint8, 1dim
				0, 0, 0x08, 2,
				// 2dmin of lengths, big endian
				0, 0, 0, 3,
				0, 0, 0, 4,
				// 3x1 values (mismatched)
				1, 2, 3,
			},
			errorContains: "EOF",
		},
	} {
		t.Run(fmt.Sprintf("case%d", idx), func(t *testing.T) {
			i, err := Read(bytes.NewReader(tc.input))
			require.Error(t, err)
			assert.Nil(t, i)
			if tc.errorContains != "" {
				assert.Contains(t, err.Error(), tc.errorContains)
			}
		})
	}
}
