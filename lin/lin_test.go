package lin

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestVectorPairOperations(t *testing.T) {
	for i, tc := range []struct {
		// Inputs
		a Vector
		b Vector
		// Expectations
		dotProduct  float32
		subtract    Vector
		elemProduct Vector
		panics      bool
	}{
		{
			a:           Vector{0, 0},
			b:           Vector{0, 0},
			dotProduct:  0,
			subtract:    Vector{0, 0},
			elemProduct: Vector{0, 0},
		},
		{
			a:           Vector{0, 1},
			b:           Vector{1, 0},
			dotProduct:  0,
			subtract:    Vector{-1, 1},
			elemProduct: Vector{0, 0},
		},
		{
			a:           Vector{1, 1},
			b:           Vector{1, 1},
			dotProduct:  2,
			subtract:    Vector{0, 0},
			elemProduct: Vector{1, 1},
		},
		{
			a:           Vector{1.5, 1},
			b:           Vector{1, 2},
			dotProduct:  3.5,
			subtract:    Vector{0.5, -1},
			elemProduct: Vector{1.5, 2},
		},
		{
			a:      Vector{1},
			b:      Vector{1, 1},
			panics: true,
		},
	} {
		t.Run(fmt.Sprintf("case%d", i), func(t *testing.T) {
			if tc.panics {
				require.Panics(t, func() {
					_ = DotProduct(tc.a, tc.b)
				})
				return
			}

			dp := DotProduct(tc.a, tc.b)
			assert.Equal(t, tc.dotProduct, dp, "dot product")

			sb := tc.a.Subtract(tc.b)
			assert.Equal(t, tc.subtract, sb, "subtract")

			ep := tc.a.ElementwiseProduct(tc.b)
			assert.Equal(t, tc.elemProduct, ep, "elementwise product")
		})
	}
}

func TestVectorScalarOperations(t *testing.T) {
	for i, tc := range []struct {
		// Inputs
		a Vector
		s float32
		// Expectations
		scalar Vector
	}{
		{
			a:      Vector{0, 0},
			s:      3.5,
			scalar: Vector{0, 0},
		},
		{
			a:      Vector{2, 5},
			s:      0,
			scalar: Vector{0, 0},
		},
		{
			a:      Vector{2, 4},
			s:      -2.5,
			scalar: Vector{-5, -10},
		},
	} {
		t.Run(fmt.Sprintf("case%d", i), func(t *testing.T) {
			ep := tc.a.Scalar(tc.s)
			assert.Equal(t, tc.scalar, ep, "scalar")
		})
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
