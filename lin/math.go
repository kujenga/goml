package lin

import "math"

// Sigmoid applies the sigmoid function to the given value.
//
// 1 / ( 1 + e^x )
func Sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

// SigmoidDerivative applies the derivative of the sigmoid function to the
// given value.
//
// sigmoid(x) * 1 - sigmoid(x)
func SigmoidDerivative(x float32) float32 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}
