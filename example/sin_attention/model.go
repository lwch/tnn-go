package main

import (
	"runtime"

	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type model struct {
	attn        []*transformer
	flatten     *layer.Flatten
	sigmoid     *activation.Sigmoid
	outputLayer *layer.Linear
	optimizer   optimizer.Optimizer
}

func newModel() *model {
	var m model
	for i := 0; i < transformerSize; i++ {
		m.attn = append(m.attn, newTransformer())
	}
	m.flatten = layer.NewFlatten("flatten")
	m.sigmoid = activation.NewSigmoid()
	m.outputLayer = layer.NewLinear("output", unitSize, 1, layer.WithDevice(device))
	m.optimizer = optimizer.NewAdam(m.params(), optimizer.WithAdamLr(lr))
	return &m
}

func (m *model) Forward(x *tensor.Tensor, train bool) *tensor.Tensor {
	y := x
	for _, attn := range m.attn {
		y = attn.Forward(y, train)
	}
	y = m.flatten.Forward(y)
	y = m.sigmoid.Forward(y)
	y = m.outputLayer.Forward(y)
	return y
}

func (m *model) Train(x, y *tensor.Tensor) {
	pred := m.Forward(x, true)
	l := lossFunc(pred, y)
	l.Backward()
	runtime.GC()
}

func (m *model) Apply() {
	m.optimizer.Step()
}

func (m *model) Predict(x *tensor.Tensor) []float32 {
	return m.Forward(x, false).Float32Value()
}

func (m *model) Loss(x, y *tensor.Tensor) float32 {
	pred := m.Forward(x, false)
	return lossFunc(pred, y).Float32Value()[0]
}

func (m *model) params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	for _, attn := range m.attn {
		ret = append(ret, attn.params()...)
	}
	for _, p := range m.outputLayer.Params() {
		ret = append(ret, p)
	}
	return ret
}
