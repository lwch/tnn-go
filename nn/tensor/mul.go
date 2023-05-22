package tensor

import (
	"gonum.org/v1/gonum/mat"
)

type mul struct {
	a, b *Tensor
}

func (op *mul) Forward() *Tensor {
	var value mat.Dense
	value.Mul(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *mul) Backward(grad *Tensor) {
	var da, db mat.Dense
	da.Mul(grad.Value(), op.b.Value().T())
	db.Mul(op.a.Value().T(), grad.Value())
	op.a.grad = FromDense(&da)
	op.b.grad = FromDense(&db)
	op.a.Backward(op.a.grad)
	op.b.Backward(op.b.grad)
}

func (op *mul) Dims() (int, int) {
	rows, _ := op.a.Value().Dims()
	_, cols := op.b.Value().Dims()
	return rows, cols
}

type mulElem struct {
	a, b *Tensor
}

func (op *mulElem) Forward() *Tensor {
	var value mat.Dense
	value.MulElem(op.a.Value(), op.b.Value())
	return FromDense(&value)
}

func (op *mulElem) Backward(grad *Tensor) {
	var da, db mat.Dense
	da.MulElem(op.b.Value(), grad.Value())
	db.MulElem(op.a.Value(), grad.Value())
	op.a.grad = FromDense(&da)
	op.b.grad = FromDense(&db)
	op.a.Backward(op.a.grad)
	op.b.Backward(op.b.grad)
}

func (op *mulElem) Dims() (int, int) {
	return op.a.Value().Dims()
}