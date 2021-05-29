// ann provides an implementation of an Artificial Neural Network
package ann

import (
	"errors"
	"fmt"
	"math/rand"
)

type ANN struct {
	LearningRate float32
	Layers       []*Layer
	Introspect   func(step Step)
}

type Step struct {
	TotalError float32
	Loss       float32
}

func (n *ANN) Train(
	epochs int,
	inputs Frame,
	labels Frame,
) error {
	// Correctness checks
	if err := n.check(inputs, labels); err != nil {
		return err
	}

	// Initialize layers
	var prev *Layer
	for _, layer := range n.Layers {
		layer.initialize(prev)
		prev = layer
	}

	// Training epochs, running against all inputs in a single batch.
	for e := 0; e < epochs; e++ {
		// Iterate FORWARDS through the network
		activations := inputs
		for _, layer := range n.Layers {
			activations = layer.ForwardProp(activations)
		}
		pred := activations

		errors := errorAmount(pred, labels)
		loss := Loss(pred, labels)
		if n.Introspect != nil {
			var errSum float32
			errors.ForEach(func(v float32) {
				errSum += v
			})
			n.Introspect(Step{
				TotalError: errSum,
				Loss:       loss,
			})
		}

		dcost := errors.DeepCopy()
		dpred := pred.DeepCopy()
		dpred.Apply(sigmoidDerivative)
		zDel := dcost.Pairwise(dpred, func(c, p float32) float32 {
			return c * p
		})
		fmt.Println("zDel:", zDel)
	}

	return nil
}

// Costs as the raw error
func errorAmount(outputs, labels Frame) Frame {
	return outputs.Pairwise(labels, func(o, l float32) float32 {
		return o - l
	})
}

func (n *ANN) check(inputs Frame, outputs Frame) error {
	if len(n.Layers) == 0 {
		return errors.New("ann must have at least one layer")
	}

	if len(inputs) != len(outputs) {
		return fmt.Errorf(
			"inputs count %d mismatched with outputs count %d",
			len(inputs), len(outputs),
		)
	}
	return nil
}

// Layer defines a layer in the neural network. Things a layer can do:
// - Transform input elements in-place
// -
type Layer struct {
	Name        string
	Width       int
	InitialBias float32

	prev *Layer

	initialized bool
	// weights are row x column. each row is a node in the current layer,
	// each column corresponds with a node from the previous layer.
	weights Frame
	// each node has a bias which can be changed over time.
	biases Vector
}

// Initialize random weights for the layer.
func (l *Layer) initialize(prev *Layer) {
	if l.initialized || prev == nil {
		// If already initialized or the input layer
		return
	}
	l.prev = prev
	l.weights = make(Frame, l.Width)
	for i := range l.weights {
		l.weights[i] = make(Vector, l.prev.Width)
		for j := range l.weights[i] {
			l.weights[i][j] = rand.Float32()
		}
	}
	l.biases = make(Vector, l.Width)
	for i := range l.biases {
		l.biases[i] = rand.Float32()
	}
}

// Takes in the values, where "inputs" is the output of the previous layer.
func (l *Layer) ForwardProp(inputs Frame) Frame {
	// If this is the input layer, there is no feed forward step.
	if l.prev == nil {
		return inputs
	}

	activations := make(Frame, len(inputs))
	// Feed forward each input through this layer.
	for i, input := range inputs {
		activations[i] = make(Vector, l.Width)
		for j := range activations[i] {
			// Find the dot product and apply bias
			z := DotProduct(input, l.weights[j]) + l.biases[j]
			// Sigmoid activation function
			activations[i][j] = sigmoid(z)
		}
	}
	return activations
}
