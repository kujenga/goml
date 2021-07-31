package mnist

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRead(t *testing.T) {
	dataset, err := Read("../testdata/mnist")
	require.NoError(t, err)
	require.NotNil(t, dataset)

	require.NotNil(t, dataset.TrainInputs)
	require.NotNil(t, dataset.TrainLabels)
	require.NotNil(t, dataset.TestInputs)
	require.NotNil(t, dataset.TestLabels)
}
