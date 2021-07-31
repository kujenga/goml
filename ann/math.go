package ann

import (
	"math"

	"github.com/kujenga/goml/lin"
)

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

func sigmoidDerivative(x float32) float32 {
	return sigmoid(x) * (1 - sigmoid(x))
}

// Loss function, mean squared error.
//
// Mean(Error^2)
func Loss(pred, labels lin.Frame) float32 {
	var squaredError, count float32
	pred.ForEachPairwise(labels, func(o, l float32) {
		count += 1.0
		// squared error
		squaredError += (o - l) * (o - l)
	})
	return squaredError / count
}
