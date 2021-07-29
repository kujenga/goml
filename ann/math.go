package ann

import (
	"fmt"
	"math"
)

type Frame []Vector

func (f Frame) DeepCopy() Frame {
	out := make(Frame, len(f))
	for i := range f {
		out[i] = make(Vector, len(f[i]))
		copy(out[i], f[i])
	}
	return out
}

func (f Frame) Apply(fn func(float32) float32) {
	for i := range f {
		for j := range f[i] {
			f[i][j] = fn(f[i][j])
		}
	}
}

func (f Frame) ForEach(fn func(float32)) {
	for i := range f {
		for j := range f[i] {
			fn(f[i][j])
		}
	}
}

func (f Frame) ForEachPairwise(o Frame, fn func(float32, float32)) {
	for i := range f {
		for j := range f[i] {
			fn(f[i][j], o[i][j])
		}
	}
}

func (f Frame) Pairwise(o Frame, fn func(float32, float32) float32) Frame {
	out := f.DeepCopy()
	for i := range f {
		for j := range f[i] {
			out[i][j] = fn(f[i][j], o[i][j])
		}
	}
	return out
}

type Vector []float32

func DotProduct(a, b Vector) float32 {
	if len(a) != len(b) {
		panic(fmt.Errorf(
			"cannot dot product arrays of unequal length: %d, %d",
			len(a),
			len(b),
		))
	}
	var res float32
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

func sigmoidDerivative(x float32) float32 {
	return sigmoid(x) * (1 - sigmoid(x))
}

// Loss function, mean squared error.
//
// Mean(Error^2)
func Loss(pred, labels Frame) float32 {
	var squaredError, count float32
	pred.ForEachPairwise(labels, func(o, l float32) {
		count += 1.0
		// squared error
		squaredError += (o - l) * (o - l)
	})
	return squaredError / count
}
