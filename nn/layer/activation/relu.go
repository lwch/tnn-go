package activation

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
)

type ReLU struct {
	*base
}

func NewReLU() *ReLU {
	var layer ReLU
	layer.base = new("relu")
	return &layer
}

func LoadRelu(name string, _ []*tensor.Tensor, _ map[string]float32) layer.Layer {
	var layer ReLU
	layer.base = new("relu")
	layer.name = name
	return &layer
}

func (layer *ReLU) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Relu()
}
