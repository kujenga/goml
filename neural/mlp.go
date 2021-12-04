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

// This work was based on learnings from the following resources:
// - "Make Your Own Neural Network" by Tariq Rashid
// - deeplizard series on "Backpropagation explained":
//   https://www.youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU
// - "Build an Artificial Neural Network From Scratch" article:
//   https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html

// MLP provides a Multi-Layer Perceptron which can be configured for
// any network architecture within that paradigm.
type MLP struct {
	// Layers is a list of layers in the network, where the first is the
	// input and last is the output, with inner layers acting as hidden
	// layers.
	//
	// These must not be modified after initialization/training.
	Layers []*Layer

	// LearningRate is the rate at which learning occurs in back
	// propagation, relative to the error calculations.
	LearningRate float32

	// Introspect provides a way for the caller of this network to
	// check the status of network learning over time and witness
	// convergence (or lack thereof).
	Introspect func(step Step)
}

// Step captures status updates that happens within a single Epoch, for use in
// introspecting models.
type Step struct {
	// Monotonically increasing counter of which training epoch this step
	// represents.
	Epoch int
	// Loss is the sum of normalized error values to for the epoch using
	// the loss function for the network.
	Loss float32
}

// Initialize sets up network layers with the needed memory allocations and
// references for proper operation. It is called automatically during training,
// provided separately only to facilitate more precise use of the network from
// a performance analysis perspective.
func (n *MLP) Initialize() {
	var prev *Layer
	for i, layer := range n.Layers {
		var next *Layer
		if i < len(n.Layers)-1 {
			next = n.Layers[i+1]
		}
		// Idempotent initialization of the layer, passing in the
		// previous and next layers for reference in training.
		layer.initialize(n, prev, next)
		prev = layer
	}
}

// Train takes in a set of inputs and a set of labels and trains the network
// using backpropagation to adjust internal weights to minimize loss, over the
// specified number of epochs. The final loss value is returned after training
// completes.
func (n *MLP) Train(epochs int, inputs, labels lin.Frame) (float32, error) {
	// Validate that the inputs match the network configuration.
	if err := n.check(inputs, labels); err != nil {
		return 0, err
	}

	// Initialize all layers within the network.
	n.Initialize()

	// Run the training process for the specified number of epochs.
	var loss float32
	for e := 0; e < epochs; e++ {
		predictions := make(lin.Frame, len(inputs))

		// Iterate over each inputs to train in SGD fashion.
		for i, input := range inputs {
			// Iterate FORWARDS through the network.
			activations := input
			for _, layer := range n.Layers {
				activations = layer.ForwardProp(activations)
			}
			predictions[i] = activations

			// Iterate BACKWARDS through the network.
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

		// Calculate loss
		loss = Loss(predictions, labels)
		if n.Introspect != nil {
			n.Introspect(Step{
				Epoch: e,
				Loss:  loss,
			})
		}

	}

	return loss, nil
}

// Predict takes in a set of input rows with the width of the input layer, and
// returns a frame of prediction rows with the width of the output layer,
// representing the predictions of the network.
func (n *MLP) Predict(inputs lin.Frame) lin.Frame {
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

func (n *MLP) check(inputs lin.Frame, outputs lin.Frame) error {
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

// Layer defines a layer in the neural network. These are presently basic
// feed-forward layers that also provide capabilities to facilitate
// backpropagatin within the MLP structure.
type Layer struct {
	// Name provides a human displayable name for the layer, for debugging
	// and documentation purposes.
	Name string
	// Width defines the number of neurons in this layer.
	Width int
	// Activation function for transforming values passed out of this
	// layer. Defaults to sigmoid when not specified.
	ActivationFunction func(float32) float32
	// Derivative of the activation function, needed for backpropagation.
	// Must match the value of the ActivationFunction variable. Defaults to
	// the derivative of the default sigmoid function.
	ActivationFunctionDeriv func(float32) float32

	// Pointer to the neural network that this layer is being used within.
	nn *MLP
	// Pointer to previous layer in the network for use in initialization
	// steps and backprop.
	prev *Layer
	// Pointer to next layer in the network for use in backprop.
	next *Layer

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
	// As backpropagation progresses, we record the value of the Loss last
	// seen so that it can be incorporated as a proxy error for the errors
	// computed in the output layer.
	lastE lin.Vector
	lastL lin.Frame
}

// initialize sets up the needed data structures and random initial values for
// the layer. If key values are unspecified, defaults are configured.
func (l *Layer) initialize(nn *MLP, prev *Layer, next *Layer) {
	if l.initialized || prev == nil {
		// If already initialized or the input layer, nothing to do.
		return
	}

	// Pointers to other components in the network.
	l.nn = nn
	l.prev = prev
	l.next = next

	// Provide defaults for unspecified hyperparameters.
	if l.ActivationFunction == nil {
		l.ActivationFunction = lin.Sigmoid
	}
	if l.ActivationFunctionDeriv == nil {
		l.ActivationFunctionDeriv = lin.SigmoidDerivative
	}

	// Initialize data structures for use in the network training and
	// predictions, providing them with random initial values.
	l.weights = make(lin.Frame, l.Width)
	for i := range l.weights {
		l.weights[i] = make(lin.Vector, l.prev.Width)
		for j := range l.weights[i] {
			// We scale these based on the "connectedness" of the
			// node to avoid saturating the gradients in the
			// network, where really high values do not play nicely
			// with activation functions like sigmoid.
			weight := rand.NormFloat64() *
				math.Pow(float64(l.prev.Width), -0.5)
			l.weights[i][j] = float32(weight)
		}
	}
	l.biases = make(lin.Vector, l.Width)
	for i := range l.biases {
		l.biases[i] = rand.Float32()
	}
	// Set up empty error and loss structures for use in backprop.
	l.lastE = make(lin.Vector, l.Width)
	l.lastL = make(lin.Frame, l.Width)
	for i := range l.lastL {
		l.lastL[i] = make(lin.Vector, l.prev.Width)
	}

	l.initialized = true
}

// ForwardProp takes in a set of inputs from the previous layer and performs
// forward propagation for the current layer, returning the resulting
// activations. As a special case, if this Layer has no previous layer and is
// thus the input layer for the network, the values are passed through
// unmodified. Internal state from the calculation is persisted for later use
// in back propagation.
func (l *Layer) ForwardProp(input lin.Vector) lin.Vector {
	// If this is the input layer, pass through values unmodified.
	if l.prev == nil {
		l.lastActivations = input
		return input
	}

	// Create vectors with state for each node in this layer.
	Z := make(lin.Vector, l.Width)
	activations := make(lin.Vector, l.Width)
	// For each node in the layer, perform feed-forward calculation.
	for i := range activations {
		// Vector of weights for each edge to this node, incoming from
		// the previous layer.
		nodeWeights := l.weights[i]
		// Scalar bias value for the current node index.
		nodeBias := l.biases[i]
		// Combine input with incoming edge weights, then apply bias.
		Z[i] = lin.DotProduct(input, nodeWeights) + nodeBias
		// Apply activation function for non-linearity.
		activations[i] = l.ActivationFunction(Z[i])
	}
	// Capture state for use in back-propagation.
	l.lastZ = Z
	l.lastActivations = activations
	return activations
}

// BackProp performs the training process of back propagation on the layer for
// the given set of labels. Weights and biases are updated for this layer
// according to the computed error. Internal state on the backpropagation
// process is captured for further backpropagation in earlier layers of the
// network as well.
func (l *Layer) BackProp(label lin.Vector) {
	// Iterate over weights for each node "j" in the current layer, where
	// "wj" is a vector of the weights for edges incoming to that node from
	// the previous layer.
	for j, wj := range l.weights {
		// ∂L/∂a, deriv Loss w.r.t. activation:
		// 2 ( a1(L) - y1 )
		// First calculate the last observed error value for node j.
		if l.next == nil { // Output layer
			// Difference between label and output value.
			l.lastE[j] = l.lastActivations[j] - label[j]
		} else {
			// Formula for propagated error in hidden layers:
			// ∑0-j ( 2(aj-yj) (g'(zj)) (wj2) )

			// Iterate over each node in the next layer down and
			// sum up the losses attributed to this node.
			l.lastE[j] = 0
			for jn := range l.next.lastL {
				// Add the loss from node jn in the next layer
				// that came from node j in this layer.
				l.lastE[j] += l.next.lastL[jn][j]
			}
		}
		// Derivative of the squared error w.r.t. activation.
		dLdA := 2 * l.lastE[j]

		// ∂a/∂z, derivative of activation w.r.t. input:
		// g'(L)(z) ( z1(L) )
		dAdZ := l.ActivationFunctionDeriv(l.lastZ[j])

		// Capture the loss for this edge for use in the next layer up
		// of backprop. This references and feeds into the lastE
		// calculation above, used in the first derivative term.
		for k, wjk := range wj {
			l.lastL[j][k] = wjk * l.lastE[j]
		}

		// Iterate over each weight for node "k" in the previous layer
		// and update it according to the computed derivatives.
		for k, wjk := range wj {
			// ∂z/∂w, derivative of input w.r.t. weight:
			// a2(L-1)
			ak := l.prev.lastActivations[k]
			dZdW := ak

			// Total derivative, via chain rule ∂L/∂w,
			// derivative of loss w.r.t. weight
			dLdW := dLdA * dAdZ * dZdW

			// Update the weight according to the learning rate.
			l.weights[j][k] = wjk - (l.nn.LearningRate * dLdW)
		}

		// Update the bias along the gradient of the loss w.r.t. inputs.
		// ∂L/∂z = ∂L/∂a * ∂a/∂z
		l.biases[j] = l.biases[j] -
			(l.nn.LearningRate * dLdA * dAdZ)
	}
}
