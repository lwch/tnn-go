package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type varianceAxis struct {
	a     *Tensor
	axis  int
	value *Tensor
}

func (op *varianceAxis) f() *mat.Dense {
	var size int
	switch op.axis {
	case 0:
		size, _ = op.a.Dims()
	case 1:
		_, size = op.a.Dims()
	default:
		panic("invalid axis")
	}
	mean := op.a.MeanAxis(op.axis)
	op.value = op.a.Sub(mean).Pow(2).SumAxis(op.axis).Scale(1 / float64(size-1))
	return op.value.Value()
}

func (op *varianceAxis) df(grad *Tensor) {
	op.value.op.df(grad)
}

func (op *varianceAxis) ZeroGrad() {
	op.a.ZeroGrad()
}