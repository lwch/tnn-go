package layer

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

func TestAttention(t *testing.T) {
	l := NewAttention("attn", 4, 1, 0, false)
	x := tensor.ARange(1*3*4, consts.KFloat).Reshape(1, 3, 4)
	y := l.Forward(x, x, x, nil, false, true)
	fmt.Println(y.Float32Value())
}

func TestAttentionRope(t *testing.T) {
	l := NewAttention("attn", 4, 1, 0, true)
	x := tensor.ARange(1*3*4, consts.KFloat).Reshape(1, 3, 4)
	y := l.Forward(x, x, x, nil, false, true)
	fmt.Println(y.Float32Value())
}

func TestAttentionScore(t *testing.T) {
	l := NewAttention("attn", 4, 1, 0, false)
	x := tensor.ARange(1*3*4, consts.KFloat).Reshape(1, 3, 4)
	score := l.Score(x, x, x, nil, false, true)
	fmt.Println(score.Float32Value())
}

func TestXxx(*testing.T) {
	const seq = 16
	const dim = 1024
	freq := buildFreqs(consts.KCPU, 4096, dim, seq)
	data := make([]float32, seq*dim)
	for i := int64(0); i < seq*dim; i++ {
		data[i] = 1
	}
	t := tensor.FromFloat32(data, tensor.WithShapes(1, seq, 1, dim))
	shapes := t.Shapes()
	t = t.Reshape(append(shapes[:len(shapes)-1], -1, 2)...)
	t = t.ViewAsComplex()
	data = t.Mul(freq).ViewAsReal().Flatten(3, -1).Contiguous().Float32Value()
	for i := int64(0); i < seq; i++ {
		fmt.Println(data[i*dim : (i+1)*dim])
	}
}
