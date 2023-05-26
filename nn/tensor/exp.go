package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type exp struct {
	a *Tensor
}

func (op *exp) expValue() *mat.Dense {
	var value mat.Dense
	value.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, op.a.Value())
	return &value
}

func (op *exp) f() *mat.Dense {
	return op.expValue()
}

func (op *exp) df(grad *Tensor) {
	delta := op.expValue()
	delta.MulElem(grad.Value(), delta)
	op.a.AddGrad(delta)
	op.a.Backward(FromDense(delta))
}

func (op *exp) ZeroGrad() {
	op.a.ZeroGrad()
}
