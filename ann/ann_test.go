package ann

import (
	"testing"
)

func TestANNBasicPerceptron(t *testing.T) {
	var epochs = 10
	// Example from:
	// https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html
	var inputs = Frame{
		{0, 1, 0},
		{0, 0, 1},
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 1, 1},
		{0, 1, 0},
	}
	var labels = Frame{
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{1},
	}

	m := ANN{
		LearningRate: 0.05,
		Layers: []*Layer{
			// Input
			{Name: "input", Width: 3},
			// // Hidden
			{Name: "hidden", Width: 2, InitialBias: 0.5},
			// Output
			{Name: "output", Width: 1, InitialBias: 0.5},
		},
		Introspect: func(s Step) {
			t.Logf("%+v", s)
		},
	}

	err := m.Train(epochs, inputs, labels)
	if err != nil {
		t.Fatalf("error training ann: %v", err)
	}
}
