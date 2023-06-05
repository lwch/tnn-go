package activation

import (
	"github.com/lwch/tnn/nn/layer"
	"gorgonia.org/gorgonia"
)

type ReLU struct {
	*base
}

func NewReLU() layer.Layer {
	var layer ReLU
	layer.base = new("relu")
	return &layer
}

func (layer *ReLU) Forward(x *gorgonia.Node, _ bool) *gorgonia.Node {
	return gorgonia.Must(gorgonia.Rectify(x))
}
