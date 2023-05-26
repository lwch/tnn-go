package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type sqrt struct {
	a *Tensor
}

func sqrtDense(x *mat.Dense) *mat.Dense {
	var value mat.Dense
	value.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v)
	}, x)
	return &value
}

func (op *sqrt) f() *mat.Dense {
	return sqrtDense(op.a.Value())
}

func (op *sqrt) df(grad *Tensor) {
	sqrt := sqrtDense(grad.Value())
	sqrt.Scale(2, sqrt)
	delta := FromDense(sqrt).Inv()
	op.a.AddGrad(delta.Value())
	op.a.Backward(delta)
}

func (op *sqrt) ZeroGrad() {
	op.a.ZeroGrad()
}
