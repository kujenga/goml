package lin

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSigmoid(t *testing.T) {
	for _, tc := range []struct {
		in     float32
		expect float32
	}{
		// Answers pulled from:
		// https://keisan.casio.com/exec/system/15157249643425
		{
			in:     0,
			expect: 0.5,
		},
		{
			in:     1,
			expect: 0.7310585786300048792512,
		},
		{
			in:     -1,
			expect: 0.2689414213699951207488,
		},
	} {
		got := Sigmoid(tc.in)
		assert.InDelta(t, tc.expect, got, 0.0000001, "input: %v", tc.in)
	}
}

func TestSigmoidDerivative(t *testing.T) {
	for _, tc := range []struct {
		in     float32
		expect float32
	}{
		// Answers pulled from:
		// https://keisan.casio.com/exec/system/15157249643425
		{
			in:     0,
			expect: 0.25,
		},
		{
			in:     1,
			expect: 0.1966119332414818525374,
		},
		{
			in:     -1,
			expect: 0.1966119332414818525374,
		},
	} {
		got := SigmoidDerivative(tc.in)
		assert.InDelta(t, tc.expect, got, 0.0000001, "input: %v", tc.in)
	}
}
