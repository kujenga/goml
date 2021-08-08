package neural

import (
	"testing"

	"github.com/kujenga/goml/lin"
	"github.com/kujenga/goml/mnist"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	// Example inputs for testing, from:
	// https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html
	basicInputs = lin.Frame{
		{0, 1, 0},
		{0, 0, 1},
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 1, 1},
		{0, 1, 0},
	}
	basicLabels = lin.Frame{
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{1},
	}

	// Basic test cases for learning boolean logic
	boolInputs = lin.Frame{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	mustLabels = lin.Frame{
		{0},
		{0},
		{1},
		{1},
	}
	andLabels = lin.Frame{
		{0},
		{0},
		{0},
		{1},
	}
	orLabels = lin.Frame{
		{0},
		{1},
		{1},
		{1},
	}
)

func predictionTestBool(t *testing.T, m *MLP, inputs, labels lin.Frame) {
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

func predictionTestOneHot(t *testing.T, m *MLP, inputs, labels lin.Frame, threshold float32) {
	results := make([]bool, len(labels))
	for i, predictionOH := range m.Predict(inputs) {
		prediction := predictionOH.MaxVal()
		label := labels[i].MaxVal()

		results[i] = prediction == label
	}

	var score float32
	for _, r := range results {
		if r {
			score += 1.0
		}
	}
	score = score / float32(len(results))

	t.Logf("one-hot predictions score: %+v", score)
	assert.Greater(t, score, threshold,
		"should be above desired performance threshold")
}

func TestMLPSingleLayerBasic(t *testing.T) {
	var epochs = 1000

	m := MLP{
		LearningRate: 0.05,
		Layers: []*Layer{
			// Input
			{Name: "input", Width: 3},
			// Output
			{Name: "output", Width: 1},
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
	predictionTestBool(t, &m, trainInputs, trainLabels)
}

func TestMLPMultiLayerBool(t *testing.T) {
	var epochs = 500

	m := MLP{
		LearningRate: 0.1,
		Layers: []*Layer{
			// Input
			{Name: "input", Width: 2},
			// Hidden
			{Name: "hidden1", Width: 3},
			// Output
			{Name: "output", Width: 1},
		},
		Introspect: func(s Step) {
			t.Logf("%+v", s)
		},
	}

	for _, tc := range []struct {
		name   string
		inputs lin.Frame
		labels lin.Frame
	}{
		{
			name:   "must",
			inputs: boolInputs,
			labels: mustLabels,
		},
		{
			name:   "and",
			inputs: boolInputs,
			labels: andLabels,
		},
		{
			name:   "or",
			inputs: boolInputs,
			labels: orLabels,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			loss, err := m.Train(epochs, tc.inputs, tc.labels)
			require.NoError(t, err, "training error")
			assert.Less(t, loss, float32(0.1), "loss should be low")

			// While not scientifically useful, we validate that
			// the network can predict it's own training data.
			predictionTestBool(t, &m, tc.inputs, tc.labels)
		})
	}
}

func TestMLPMultiLayerBasic(t *testing.T) {
	var epochs = 300

	m := MLP{
		LearningRate: 0.1,
		Layers: []*Layer{
			// Input
			{Name: "input", Width: 3},
			// Hidden
			{Name: "hidden1", Width: 3},
			// Output
			{Name: "output", Width: 1},
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
	predictionTestBool(t, &m, trainInputs, trainLabels)
}

func TestMLPMultiLayerMNIST(t *testing.T) {
	dataset, err := mnist.Read("../testdata/mnist")
	require.NoError(t, err)

	// Training and validation

	const epochs = 5

	m := MLP{
		LearningRate: 0.1,
		Layers: []*Layer{
			// Input
			{Name: "input", Width: 28 * 28},
			// Hidden
			{Name: "hidden1", Width: 100},
			// Output
			{Name: "output", Width: 10},
		},
		Introspect: func(s Step) {
			t.Logf("%+v", s)
		},
	}

	loss, err := m.Train(epochs, dataset.TrainInputs[:10000], dataset.TrainLabels[:10000])
	require.NoError(t, err, "training error")
	assert.Less(t, loss, float32(0.1), "loss should be low")

	// Validate against the test set
	predictionTestOneHot(t, &m,
		dataset.TestInputs,
		dataset.TestLabels,
		0.9,
	)
}

func BenchmarkMLPMultiLayerBasic(b *testing.B) {
	m := MLP{
		LearningRate: 0.1,
		Layers: []*Layer{
			// Input
			{Name: "input", Width: 3},
			// Hidden
			{Name: "hidden1", Width: 3},
			// Output
			{Name: "output", Width: 1},
		},
	}

	m.Initialize()

	trainInputs := basicInputs
	trainLabels := basicLabels

	const epochs = 10

	for i := 0; i < b.N; i++ {
		_, err := m.Train(epochs, trainInputs, trainLabels)
		require.NoError(b, err, "training error")
	}
}
