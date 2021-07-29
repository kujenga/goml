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
	Epoch      int
	TotalError float32
	Loss       float32
}

// Train takes in a set of inputs and a set of labels and trains the network
// using backpropagation to adjust internal weights to minimize loss, over the
// specified number of epochs. The final loss value is returned after training
// completes.
func (n *ANN) Train(
	epochs int,
	inputs Frame,
	labels Frame,
) (float32, error) {
	// Correctness checks
	if err := n.check(inputs, labels); err != nil {
		return 0, err
	}

	// Initialize layers
	var prev *Layer
	for _, layer := range n.Layers {
		layer.initialize(prev)
		prev = layer
	}

	// Training epochs, running against all inputs in a single batch.
	var loss float32
	for e := 0; e < epochs; e++ {
		// Iterate FORWARDS through the network
		activations := inputs
		for _, layer := range n.Layers {
			activations = layer.ForwardProp(activations)
		}
		pred := activations

		errors := errorAmount(pred, labels)
		loss = Loss(pred, labels)
		if n.Introspect != nil {
			var errSum float32
			errors.ForEach(func(v float32) {
				errSum += v
			})
			n.Introspect(Step{
				Epoch:      e,
				TotalError: errSum,
				Loss:       loss,
			})
		}

		// Iterate BACKWARDS through the network
		for i := range n.Layers {
			l := len(n.Layers) - (i + 1)
			layer := n.Layers[l]

			var prev *Layer
			if l == 0 {
				// If we are at the input layer, nothing to do.
				continue
			} else {
				prev = n.Layers[l-1]
			}
			var next *Layer
			if l < len(n.Layers)-1 {
				next = n.Layers[l+1]
			}

			// Backpropagation

			// Iterate over inputs, treating them as a single batch.
			for i := range inputs {
				// Iterate over weights for each node "j" in the
				// current layer.
				for j, wj := range layer.weights {
					// ∂C/∂a, deriv Cost wrt. activation
					// 2 ( a1(L) - y1 )
					aj := layer.lastActivations[i][j]
					if next == nil {
						// Output layer, just
						// needs to consider
						// this activation
						// value.
						layer.lastE[j] = aj - labels[i][j]
					} else {
						// ∑0-j ( 2(aj-yj) (g'(zj)) (wj2) )

						// Iterate over each node in
						// the next layer and sum up
						// the costs attributed to this
						// node.
						layer.lastE[j] = 0
						for jn := range next.lastC {
							// Add the cost from
							// node jn in the next
							// layer that came from
							// node j in this
							// layer.
							layer.lastE[j] += next.lastC[jn][j]
						}
					}
					// deriv of the squared error w.r.t. activation
					dCdA := 2 * layer.lastE[j]

					// ∂a/∂z, deriv activation wrt. input
					// g'(L)(z) ( z1(L) )
					dAdZ := sigmoidDerivative(layer.lastZ[i][j])

					// Capture the cost for this edge for
					// use in the next layer up of
					// backprop.
					for k, wjk := range wj {
						layer.lastC[j][k] = wjk * layer.lastE[j]
					}

					// Iterate over each weight for node "k" in the
					// previous layer.
					for k, wjk := range wj {
						// ∂z/∂w, deriv input wrt. weight
						// a2(L-1)
						ak := prev.lastActivations[i][k]
						dZdW := ak

						// Total derivative, via chain rule
						// ∂C/∂w, deriv cost wrt. weight
						dCdW := dCdA * dAdZ * dZdW

						// Update the weight
						update := n.LearningRate * dCdW
						layer.weights[j][k] = wjk - update
					}

					// Update the bias with ∂C/∂a * ∂a/∂z
					layer.biases[j] = layer.biases[j] -
						n.LearningRate*dCdA*dAdZ
				}
			}
		}
	}

	return loss, nil
}

// Predict takes in a set of input rows with the width of the input layer, and
// returns a frame of prediction rows with the width of the output layer,
// representing the predictions of the network.
func (n *ANN) Predict(inputs Frame) Frame {
	// Iterate FORWARDS through the network
	activations := inputs
	for _, layer := range n.Layers {
		activations = layer.ForwardProp(activations)
	}
	pred := activations
	return pred
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

	// Every time that input is fed through this layer, the last input
	// values "z" computed from the weights and the last activation values are
	// preserved for use in backpropagation.
	lastZ           Frame
	lastActivations Frame
	// As backpropagation progresses, we record the value of the Cost last
	// seen so that it can be incorporated as a proxy error for the errors
	// computed in the output layer.
	lastE Vector
	lastC Frame
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
	// Setup as empty for use in backprop
	l.lastE = make(Vector, l.Width)
	l.lastC = make(Frame, l.Width)
	for i := range l.lastC {
		l.lastC[i] = make(Vector, l.prev.Width)
	}
}

// Takes in the values, where "inputs" is the output of the previous layer.
func (l *Layer) ForwardProp(inputs Frame) Frame {
	// If this is the input layer, there is no feed forward step.
	if l.prev == nil {
		l.lastActivations = inputs
		return inputs
	}

	Z := make(Frame, len(inputs))
	activations := make(Frame, len(inputs))
	// Feed forward each input through this layer.
	for i, input := range inputs {
		Z[i] = make(Vector, l.Width)
		activations[i] = make(Vector, l.Width)
		for j := range activations[i] {
			// Find the dot product and apply bias
			Z[i][j] = DotProduct(input, l.weights[j]) + l.biases[j]
			// Sigmoid activation function
			activations[i][j] = sigmoid(Z[i][j])
		}
	}
	l.lastZ = Z
	l.lastActivations = activations
	return activations
}
