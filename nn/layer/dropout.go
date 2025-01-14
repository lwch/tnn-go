package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type Dropout struct {
	base
	keep float64
}

func NewDropout(name string, keep float64) *Dropout {
	var layer Dropout
	layer.new("dropout", name)
	layer.keep = keep
	return &layer
}

func LoadDropout(name string, _ []*tensor.Tensor, args map[string]float32) Layer {
	var layer Dropout
	layer.new("dropout", name)
	layer.keep = float64(args["keep"])
	return &layer
}

func (layer *Dropout) Forward(x *tensor.Tensor, train bool) *tensor.Tensor {
	return x.Dropout(layer.keep, train)
}

func (layer *Dropout) Args() map[string]float32 {
	return map[string]float32{
		"keep": float32(layer.keep),
	}
}

func (layer *Dropout) ToScalarType(t consts.ScalarType) {
}

func (layer *Dropout) Reset() {
}
