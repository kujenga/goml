package ann

import (
	"github.com/kujenga/goml/lin"
)

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
