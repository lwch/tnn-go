package model

import (
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
)

type transformer struct {
	attn    *layer.SelfAttention
	nor     *layer.Nor
	flatten *layer.Flatten
	dense   *layer.Dense
	relu    *activation.ReLU
	output  *layer.Dense
}

func newTransformer() *transformer {
	return &transformer{
		attn:    layer.NewSelfAttention(paddingSize, embeddingDim),
		nor:     layer.NewNor(),
		flatten: layer.NewFlatten(),
		dense:   layer.NewDense(unitSize * 4),
		relu:    activation.NewReLU(),
		output:  layer.NewDense(unitSize),
	}
}

func (t *transformer) forward(x *tensor.Tensor) *tensor.Tensor {
	batchSize := x.Shapes()[0]
	y := t.attn.Forward(x)
	y = y.Add(x)
	selfOut := t.nor.Forward(y)
	y = t.flatten.Forward(y)
	y = t.dense.Forward(y)
	y = t.relu.Forward(y)
	y = t.output.Forward(y)
	y = y.Reshape(batchSize, paddingSize, embeddingDim)
	y = y.Add(selfOut)
	y = t.nor.Forward(y)
	return y
}

func (t *transformer) params() []*tensor.Tensor {
	var ret []*tensor.Tensor
	for _, p := range t.attn.Params() {
		ret = append(ret, p)
	}
	for _, p := range t.dense.Params() {
		ret = append(ret, p)
	}
	for _, p := range t.output.Params() {
		ret = append(ret, p)
	}
	return ret
}

func (t *transformer) layers() []layer.Layer {
	return []layer.Layer{
		t.attn,
		t.nor,
		t.flatten,
		t.dense,
		t.relu,
		t.output,
	}
}