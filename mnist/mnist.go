// Package mnist provides utilities for reading the MNIST dataset.
//
// For a description of this data format, see the documentation here:
// http://yann.lecun.com/exdb/mnist/
package mnist

import (
	"path/filepath"

	"github.com/kujenga/goml/idx"
	"github.com/kujenga/goml/lin"
)

// MNIST provides a parsed form of the dataset, mirroring the files distributed
// with it.
type MNIST struct {
	// Inputs for training, as flattened lists of pixels normalized (0,1)
	TrainInputs lin.Frame
	// Labels for training, as one-hot encoded digit indications.
	TrainLabels lin.Frame

	// Inputs for testing, as flattened lists of pixels normalized (0,1)
	TestInputs lin.Frame
	// Labels for testing, as one-hot encoded digit indications.
	TestLabels lin.Frame
}

// Read returns a parsed MNIST dataset from the indicated root directory. It
// expects files within that directory to be present as specified in the
// documentation: http://yann.lecun.com/exdb/mnist/
func Read(rootDir string) (*MNIST, error) {
	trainRawImages, err := idx.ReadFile(
		filepath.Join(rootDir, "train-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	trainRawLabels, err := idx.ReadFile(
		filepath.Join(rootDir, "train-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, err
	}

	testRawImages, err := idx.ReadFile(
		filepath.Join(rootDir, "t10k-images-idx3-ubyte.gz"))
	if err != nil {
		return nil, err
	}
	testRawLabels, err := idx.ReadFile(
		filepath.Join(rootDir, "t10k-labels-idx1-ubyte.gz"))
	if err != nil {
		return nil, err
	}

	out := &MNIST{
		TrainInputs: make(lin.Frame, trainRawImages.Dimensions[0]),
		TrainLabels: make(lin.Frame, trainRawLabels.Dimensions[0]),
		TestInputs:  make(lin.Frame, testRawImages.Dimensions[0]),
		TestLabels:  make(lin.Frame, testRawLabels.Dimensions[0]),
	}

	// Convert images to the desired (0, 1) format
	for i := range out.TrainInputs {
		out.TrainInputs[i] = make([]float32, 28*28)
		for j := range out.TrainInputs[i] {
			out.TrainInputs[i][j] = float32(
				trainRawImages.Data[i*28*28+j])/255.0*0.99 + 0.01
		}
	}
	for i := range out.TestInputs {
		out.TestInputs[i] = make([]float32, 28*28)
		for j := range out.TestInputs[i] {
			out.TestInputs[i][j] = float32(
				testRawImages.Data[i*28*28+j])/255.0*0.99 + 0.01
		}
	}

	// Convert labels to the desired one-hot format
	for i := range out.TrainLabels {
		out.TrainLabels[i] = make([]float32, 10)
		for j := range out.TrainLabels[i] {
			out.TrainLabels[i][j] = 0.01
		}
		out.TrainLabels[i][trainRawLabels.Data[i]] = 0.99
	}
	for i := range out.TestLabels {
		out.TestLabels[i] = make([]float32, 10)
		for j := range out.TestLabels[i] {
			out.TestLabels[i][j] = 0.01
		}
		out.TestLabels[i][testRawLabels.Data[i]] = 0.99
	}

	return out, nil
}
