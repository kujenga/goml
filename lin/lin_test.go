package lin

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFrameDeepCopy(t *testing.T) {
	a := Frame{
		{1, 2},
		{3, 4},
	}
	b := a.DeepCopy()
	assert.Equal(t, a, b)

	a[0][0] = 10
	assert.NotEqual(t, b, a)
}

func TestFrameApply(t *testing.T) {
	a := Frame{
		{1, 2},
		{3, 4},
	}

	a.Apply(func(i float32) float32 { return i*2 + 1 })

	assert.Equal(t, Frame{
		{3, 5},
		{7, 9},
	}, a)
}

func TestFrameForEach(t *testing.T) {
	a := Frame{
		{1, 1},
		{1, 1},
	}

	var acc float32
	a.ForEach(func(i float32) { acc += i })

	assert.Equal(t, (float32)(4), acc)
}

func TestFrameForEachPairwise(t *testing.T) {
	a := Frame{
		{2, 2},
	}
	b := Frame{
		{2, 2},
	}

	var acc float32
	a.ForEachPairwise(b, func(x, y float32) { acc += x * y })

	assert.Equal(t, (float32)(8), acc)
}

func TestFramePairwise(t *testing.T) {
	a := Frame{
		{2, 3},
	}
	b := Frame{
		{2, 3},
	}

	c := a.Pairwise(b, func(x, y float32) float32 { return x * y })

	assert.Equal(t, Frame{
		{4, 9},
	}, c)
}

func TestDotProduct(t *testing.T) {
	for i, tc := range []struct {
		a      []float32
		b      []float32
		answer float32
		panics bool
	}{
		{
			a:      []float32{0, 0},
			b:      []float32{0, 0},
			answer: 0,
		},
		{
			a:      []float32{0, 1},
			b:      []float32{1, 0},
			answer: 0,
		},
		{
			a:      []float32{1, 1},
			b:      []float32{1, 1},
			answer: 2,
		},
		{
			a:      []float32{1.5, 1},
			b:      []float32{1, 1.1},
			answer: 2.6,
		},
		{
			a:      []float32{1},
			b:      []float32{1, 1},
			panics: true,
		},
	} {
		if tc.panics {
			assert.Panics(t, func() {
				_ = DotProduct(tc.a, tc.b)
			})
			continue
		}

		got := DotProduct(tc.a, tc.b)
		assert.Equal(t, tc.answer, got, "case: %d, i", i)
	}
}

func TestFrameMaxVal(t *testing.T) {
	for i, tc := range []struct {
		a    Vector
		imax int
	}{
		{
			a:    Vector{},
			imax: 0,
		},
		{
			a:    Vector{4, 1, 2},
			imax: 0,
		},
		{
			a:    Vector{1, 4, 2},
			imax: 1,
		},
		{
			a:    Vector{-5, -1, -4, -2},
			imax: 1,
		},
	} {
		assert.Equal(t, tc.imax, tc.a.MaxVal(), "case %d: %v", i, tc.a)
	}
}
