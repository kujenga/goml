package ann

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Example inputs for testing, from:
// https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html
var (
	basicInputs = Frame{
		{0, 1, 0},
		{0, 0, 1},
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 1, 1},
		{0, 1, 0},
	}
	basicLabels = Frame{
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{1},
	}
)

func predictionTest(t *testing.T, m *ANN, inputs, labels Frame) {
	for i, prediction := range m.Predict(inputs) {
		t.Logf("input: %+v, prediction: %+v, label: %+v",
			inputs[i], prediction, labels[i])

		for j, val := range prediction {
			const midpoint float32 = 0.5
			if labels[i][j] < midpoint {
				assert.Less(t, val, midpoint)
			} else {
				assert.Greater(t, val, midpoint)
			}
		}
	}
}

func TestANNSingleLayer(t *testing.T) {
	var epochs = 1000

	m := ANN{
		LearningRate: 0.05,
		Layers: []*Layer{
			// Input
			{Name: "input", Width: 3},
			// Output
			{Name: "output", Width: 1, InitialBias: 0.5},
		},
		Introspect: func(s Step) {
			t.Logf("%+v", s)
		},
	}

	trainInputs := basicInputs
	trainLabels := basicLabels

	loss, err := m.Train(epochs, trainInputs, trainLabels)
	require.NoError(t, err, "training error")
	assert.Less(t, loss, float32(0.1), "loss should be low")

	// While not scientifically useful, we validate that the network can
	// predict it's own training data.
	predictionTest(t, &m, trainInputs, trainLabels)
}

func TestANNMultiLayer(t *testing.T) {
	var epochs = 1000

	m := ANN{
		LearningRate: 0.05,
		Layers: []*Layer{
			// Input
			{Name: "input", Width: 3},
			// Output
			{Name: "hidden1", Width: 2, InitialBias: 0.5},
			// Output
			{Name: "output", Width: 1, InitialBias: 0.5},
		},
		Introspect: func(s Step) {
			t.Logf("%+v", s)
		},
	}

	trainInputs := basicInputs
	trainLabels := basicLabels

	loss, err := m.Train(epochs, trainInputs, trainLabels)
	require.NoError(t, err, "training error")
	assert.Less(t, loss, float32(0.1), "loss should be low")

	// While not scientifically useful, we validate that the network can
	// predict it's own training data.
	predictionTest(t, &m, trainInputs, trainLabels)
}
