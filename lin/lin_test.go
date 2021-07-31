package lin

import "testing"

func TestDotProduct(t *testing.T) {
	for i, tc := range []struct {
		a      []float32
		b      []float32
		answer float32
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
	} {
		got := DotProduct(tc.a, tc.b)
		if got != tc.answer {
			t.Errorf("%d: got %f did not match expected %f",
				i, got, tc.answer)
		}
	}
}
