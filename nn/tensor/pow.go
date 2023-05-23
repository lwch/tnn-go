package tensor

type pow struct {
	a *Tensor
	b float64
}

func (op *pow) Forward() *Tensor {
	value := powDense(op.a.Value(), op.b)
	return FromDense(value)
}

func (op *pow) Backward(grad *Tensor) {
	pow := powDense(op.a.Value(), op.b-1)
	pow.Scale(op.b, pow)
	if op.a.grad == nil {
		op.a.grad = Zeros(pow.Dims())
	}
	op.a.grad.AddValue(pow)
	op.a.Backward(FromDense(pow))
}

func (op *pow) Dims() (int, int) {
	return op.a.Dims()
}

func (op *pow) ZeroGrad() {
	op.a.ZeroGrad()
}