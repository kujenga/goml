// Package lin implements basic linear algebra operations
package lin

import (
	"fmt"
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
