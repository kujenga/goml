// Package neural provides an implementation of an Artificial Neural Network
//
package neural

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/kujenga/goml/lin"
)

// ANN provides an Artificial Neural Network which can be configured for
// arbitrarily complex machine learning tasks.
//
// This work was based on learnings from the following resources:
// - "Make Your Own Neural Network" by Tariq Rashid
// - deeplizard series on "Backpropagation explained":
//   https://www.youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU
// - "Build an Artificial Neural Network From Scratch" article:
//   https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html
type ANN struct {
	LearningRate float32
	Layers       []*Layer
	Introspect   func(step Step)
}

// Step captures status updates that happens within a single Epoch, for use in
// introspecting models.
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
	inputs lin.Frame,
	labels lin.Frame,
) (float32, error) {
	// Correctness checks
	if err := n.check(inputs, labels); err != nil {
		return 0, err
	}

	// Initialize layers
	var prev *Layer
	for i, layer := range n.Layers {
		var next *Layer
		if i < len(n.Layers)-1 {
			next = n.Layers[i+1]
		}
		layer.initialize(n, prev, next)
		prev = layer
	}

	// Training epochs, running against all inputs in a single batch.
	var loss float32
	for e := 0; e < epochs; e++ {
		preds := make(lin.Frame, len(inputs))

		// Iterate over inputs to train in SGD fashion
		// TODO: Add batch/mini-batch options
		for i, input := range inputs {
			// Iterate FORWARDS through the network
			activations := input
			for _, layer := range n.Layers {
				activations = layer.ForwardProp(activations)
			}
			preds[i] = activations

			// Iterate BACKWARDS through the network
			for step := range n.Layers {
				l := len(n.Layers) - (step + 1)
				layer := n.Layers[l]

				if l == 0 {
					// If we are at the input layer, nothing to do.
					continue
				}

				layer.BackProp(labels[i])
			}
		}

		// Calculate errors
		errors := errorAmount(preds, labels)
		loss = Loss(preds, labels)
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

	}

	return loss, nil
}

// Predict takes in a set of input rows with the width of the input layer, and
// returns a frame of prediction rows with the width of the output layer,
// representing the predictions of the network.
func (n *ANN) Predict(inputs lin.Frame) lin.Frame {
	// Iterate FORWARDS through the network
	preds := make(lin.Frame, len(inputs))
	for i, input := range inputs {
		activations := input
		for _, layer := range n.Layers {
			activations = layer.ForwardProp(activations)
		}
		// Activations from the last layer are our predictions
		preds[i] = activations
	}
	return preds
}

// Costs as the raw error
func errorAmount(outputs, labels lin.Frame) lin.Frame {
	return outputs.Pairwise(labels, func(o, l float32) float32 {
		return o - l
	})
}

func (n *ANN) check(inputs lin.Frame, outputs lin.Frame) error {
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
	Name  string
	Width int

	// Pointer to the neural network that this layer is being used within.
	nn *ANN
	// Pointer to previous layer in the network for use in initialization
	// steps and backprop.
	prev *Layer
	// Pointer to next layer in the network for use in backprop.
	next *Layer

	// Activation function, defaults to sigmoid when not specified.
	ActivationFunction func(float32) float32
	// Derivative of the activation function, defaults to the derivative of
	// the default sigmoid function.
	ActivationFunctionDeriv func(float32) float32

	initialized bool
	// weights are row x column. each row is a node in the current layer,
	// each column corresponds with a node from the previous layer.
	weights lin.Frame
	// each node has a bias which can be changed over time.
	biases lin.Vector

	// Every time that input is fed through this layer, the last input
	// values "z" computed from the weights and the last activation values are
	// preserved for use in backpropagation.
	lastZ           lin.Vector
	lastActivations lin.Vector
	// As backpropagation progresses, we record the value of the Cost last
	// seen so that it can be incorporated as a proxy error for the errors
	// computed in the output layer.
	lastE lin.Vector
	lastC lin.Frame
}

// Initialize random weights for the layer.
func (l *Layer) initialize(nn *ANN, prev *Layer, next *Layer) {
	if l.initialized || prev == nil {
		// If already initialized or the input layer
		return
	}

	// Pointers to other components in the network
	l.nn = nn
	l.prev = prev
	l.next = next

	// Hyperparameters for how the layer behaves.
	if l.ActivationFunction == nil {
		l.ActivationFunction = lin.Sigmoid
	}
	if l.ActivationFunctionDeriv == nil {
		l.ActivationFunctionDeriv = lin.SigmoidDerivative
	}

	// Memory structures for use in the network training and predictions.
	l.weights = make(lin.Frame, l.Width)
	for i := range l.weights {
		l.weights[i] = make(lin.Vector, l.prev.Width)
		for j := range l.weights[i] {
			// We scale this based on the "connectedness" of the
			// node to avoid saturating the gradients in the network.
			weight := rand.NormFloat64() * math.Pow(float64(l.prev.Width), -0.5)
			l.weights[i][j] = float32(weight)
		}
	}
	l.biases = make(lin.Vector, l.Width)
	for i := range l.biases {
		l.biases[i] = rand.Float32()
	}
	// Setup as empty for use in backprop
	l.lastE = make(lin.Vector, l.Width)
	l.lastC = make(lin.Frame, l.Width)
	for i := range l.lastC {
		l.lastC[i] = make(lin.Vector, l.prev.Width)
	}

	l.initialized = true
}

// ForwardProp takes in the values, where "inputs" is the output of the
// previous layer, and performs forward propagation.
func (l *Layer) ForwardProp(input lin.Vector) lin.Vector {
	// If this is the input layer, there is no feed forward step.
	if l.prev == nil {
		l.lastActivations = input
		return input
	}

	Z := make(lin.Vector, l.Width)
	activations := make(lin.Vector, l.Width)
	// Feed forward each input through this layer.
	for i := range activations {
		// Find the dot product and apply bias
		Z[i] = lin.DotProduct(input, l.weights[i]) + l.biases[i]
		// Sigmoid activation function
		activations[i] = l.ActivationFunction(Z[i])
	}
	l.lastZ = Z
	l.lastActivations = activations
	return activations
}

// BackProp performs back propagation for the given set of labels, updating the
// weights for this layer according to the computed error.
func (l *Layer) BackProp(label lin.Vector) {
	// Iterate over weights for each node "j" in the
	// current layer.
	for j, wj := range l.weights {
		// ∂C/∂a, deriv Cost w.r.t. activation 2 ( a1(L) - y1 )
		aj := l.lastActivations[j]
		if l.next == nil {
			// Output layer, just needs to consider this activation
			// value.
			l.lastE[j] = aj - label[j]
		} else {
			// ∑0-j ( 2(aj-yj) (g'(zj)) (wj2) )

			// Iterate over each node in the next layer and sum up
			// the costs attributed to this node.
			l.lastE[j] = 0
			for jn := range l.next.lastC {
				// Add the cost from node jn in the next layer
				// that came from node j in this layer.
				l.lastE[j] += l.next.lastC[jn][j]
			}
		}
		// deriv of the squared error w.r.t. activation
		dCdA := 2 * l.lastE[j]

		// ∂a/∂z, deriv activation w.r.t. input
		// g'(L)(z) ( z1(L) )
		dAdZ := l.ActivationFunctionDeriv(l.lastZ[j])

		// Capture the cost for this edge for use in the next layer up
		// of backprop.
		for k, wjk := range wj {
			l.lastC[j][k] = wjk * l.lastE[j]
		}

		// Iterate over each weight for node "k" in the previous layer.
		for k, wjk := range wj {
			// ∂z/∂w, deriv input w.r.t. weight a2(L-1)
			ak := l.prev.lastActivations[k]
			dZdW := ak

			// Total derivative, via chain rule ∂C/∂w,
			// deriv cost w.r.t. weight
			dCdW := dCdA * dAdZ * dZdW

			// Update the weight
			update := l.nn.LearningRate * dCdW
			l.weights[j][k] = wjk - update
		}

		// Update the bias along the gradient of the cost w.r.t. inputs.
		// ∂C/∂z = ∂C/∂a * ∂a/∂z
		l.biases[j] = l.biases[j] -
			l.nn.LearningRate*dCdA*dAdZ
	}
}
