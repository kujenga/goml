// Package lin implements basic linear algebra operations.
package lin

import (
	"fmt"
)

// Frame is a 2D metrix for use in linear algebra.
type Frame []Vector

// DeepCopy creates a copy of the Frame with no shared memory from the
// original.
func (f Frame) DeepCopy() Frame {
	out := make(Frame, len(f))
	for i := range f {
		out[i] = make(Vector, len(f[i]))
		copy(out[i], f[i])
	}
	return out
}

// Apply takes the specified function and applies it to each element in the
// Frame, modifying it in-place.
func (f Frame) Apply(fn func(float32) float32) {
	for i := range f {
		for j := range f[i] {
			f[i][j] = fn(f[i][j])
		}
	}
}

// ForEach runs the specified function against element in the Frame.
func (f Frame) ForEach(fn func(float32)) {
	for i := range f {
		for j := range f[i] {
			fn(f[i][j])
		}
	}
}

// ForEachPairwise runs the specified function against each pair of elements in
// the two frames.
func (f Frame) ForEachPairwise(o Frame, fn func(float32, float32)) {
	for i := range f {
		for j := range f[i] {
			fn(f[i][j], o[i][j])
		}
	}
}

// Pairwise creates a new frame based on the result of running the specified
// function against each pair of matching elements from within the two frames.
func (f Frame) Pairwise(o Frame, fn func(float32, float32) float32) Frame {
	out := f.DeepCopy()
	for i := range f {
		for j := range f[i] {
			out[i][j] = fn(f[i][j], o[i][j])
		}
	}
	return out
}

// Vector is a 1D array of values for use in linear algebra computations.
type Vector []float32

// DotProduct returns the value of the dot product for two vectors.
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

// MaxVal returns the index of the maximum value within the Vector. This is
// useful with interpreting the output of one-hot encoding models.
func (v Vector) MaxVal() int {
	var max float32
	var imax int
	for i, val := range v {
		if val > max {
			max = val
			imax = i
		}
	}
	return imax
}
