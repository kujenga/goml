// Package idx implements the idx data format.
//
// For a description of this data format, see the documentation here:
// http://yann.lecun.com/exdb/mnist/
package idx

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// IDX provides the basic parsed content of an IDX file. The meaning and
// structure of these fields comes from: http://yann.lecun.com/exdb/mnist/
//
// Currently, only uint8 formatted data is supported (used by the MNIST
// dataset).
type IDX struct {
	Dimensions []uint32
	Data       []uint8
}

// ReadFile returns a parsed IDX file from the specified path. If the file has
// a ".gz" extension on the end, it will assume that it is gzip compressed and
// decompress it on the fly.
func ReadFile(filename string) (*IDX, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var rdr io.Reader = f
	// Optionally decompress gzipped files on the fly.
	if filepath.Ext(filename) == ".gz" {
		gzrdr, err := gzip.NewReader(rdr)
		if err != nil {
			return nil, err
		}
		rdr = gzrdr
	}

	return Read(rdr)
}

// Read returns a parsed IDX file from the specified reader.
func Read(rdr io.Reader) (*IDX, error) {
	// Read in the "magic number" which provides us with information about
	// the format of the data in this IDX file.
	magic := make([]byte, 4)
	n, err := rdr.Read(magic)
	if err != nil {
		return nil, err
	}
	if n != len(magic) {
		return nil, fmt.Errorf("read %d bytes for magic number, expected 4", n)
	}

	// Check the first two bytes are zero, per the spec
	if magic[0] != 0 || magic[1] != 0 {
		return nil, fmt.Errorf(
			"first two bytes of magic number should be zero, got: %q", magic)
	}

	// Check that the data type is uint8, which is what we are setup for.
	if magic[2] != 0x08 {
		return nil, fmt.Errorf(
			"only uint8 data type supported, got: %x", magic[2])
	}

	// Number of dimensions of the data
	ndim := magic[3]

	o := &IDX{
		Dimensions: make([]uint32, ndim),
	}

	// Read the sizes of each dimension
	// > sizes in each dimension are 4-byte integers (MSB first, high endian...
	if err := binary.Read(rdr, binary.BigEndian, &o.Dimensions); err != nil {
		return nil, err
	}

	var totalLen uint32 = 1
	for _, d := range o.Dimensions {
		totalLen = totalLen * d
	}

	// Read in the remaining values from the file.
	o.Data = make([]uint8, int(totalLen))
	if err := binary.Read(rdr, binary.BigEndian, &o.Data); err != nil {
		return nil, err
	}

	return o, nil
}
