package layer

import (
	"math"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/tnn/internal/pb"
)

type Attention1 struct {
	base
	dims, heads int
	dropout     float64
	// params
	w     *tensor.Tensor
	b     *tensor.Tensor
	scale *tensor.Tensor
}

func NewAttention1(dims, heads int, dropout float64, opts ...LayerCreateOption) *Attention1 {
	var layer Attention1
	layer.new("attention1", opts...)
	layer.dims = dims
	layer.heads = heads
	layer.dropout = dropout
	if layer.dims%layer.heads != 0 {
		panic("dims must be divisible by heads")
	}
	layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(dims)))},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	layer.w = layer.initW(int64(dims*3), int64(dims*3))
	layer.b = layer.initB(int64(dims * 3))
	return &layer
}

func LoadAttention1(device consts.DeviceType, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Attention1
	layer.new("attention1", WithDevice(device))
	layer.name = name
	layer.dims = int(args["dims"])
	layer.heads = int(args["heads"])
	layer.dropout = float64(args["dropout"])
	layer.w = layer.loadParam(params["w"])
	layer.b = layer.loadParam(params["b"])
	layer.scale = tensor.FromFloat32(nil, []float32{float32(math.Sqrt(float64(layer.dims)))},
		tensor.WithShapes(1),
		tensor.WithDevice(layer.device))
	return &layer
}

func (layer *Attention1) Forward(q, k, v, mask *tensor.Tensor, train, isCausal bool) (*tensor.Tensor, *tensor.Tensor) {
	if mask != nil && isCausal {
		panic("unexpected mask")
	}
	inputShape := q.Shapes()
	x := tensor.Cat([]*tensor.Tensor{q, k, v}, -1)           // (batch, seq, dims*3)
	x = x.MatMul(layer.w).Add(layer.b)                       // (batch, seq, dims*3)
	q = x.NArrow(-1, 0, int64(layer.dims))                   // (batch, seq, dims)
	k = x.NArrow(-1, int64(layer.dims), int64(layer.dims))   // (batch, seq, dims)
	v = x.NArrow(-1, int64(layer.dims*2), int64(layer.dims)) // (batch, seq, dims)
	q = layer.split(q)                                       // (batch, heads, seq, dims/heads)
	k = layer.split(k)                                       // (batch, heads, seq, dims/heads)
	v = layer.split(v)                                       // (batch, heads, seq, dims/heads)
	if isCausal {
		mask = layer.buildCausal(q, k)
	}
	score := q.MatMul(k.Transpose(-1, -2)).Div(layer.scale) // (batch, heads, seq, dims/heads)
	if mask != nil {
		score = score.Add(mask) // (batch, heads, ..., dims/heads)
	}
	score = score.Softmax1(-1)                          // (batch, heads, seq, dims/heads)
	score = score.Dropout(layer.dropout, train)         // (batch, heads, seq, dims/heads)
	y := score.MatMul(v)                                // (batch, heads, seq, dims/heads)
	y = y.Transpose(1, 2)                               // (batch, seq, heads, dims/heads)
	y = y.Reshape(-1, inputShape[1], int64(layer.dims)) // (batch, seq, dims)
	return y, score
}

func (layer *Attention1) split(x *tensor.Tensor) *tensor.Tensor {
	y := x.View(-1, x.Shapes()[1], int64(layer.heads), int64(layer.dims/layer.heads))
	return y.Transpose(1, 2)
}

func (layer *Attention1) buildCausal(q, k *tensor.Tensor) *tensor.Tensor {
	l := q.Shapes()[q.Dims()-2]
	s := k.Shapes()[k.Dims()-2]
	mask := make([]float32, l*s)
	for i := int64(0); i < l; i++ {
		for j := int64(0); j < s; j++ {
			if j > i {
				mask[i*s+j] = float32(math.Inf(-1))
			}
		}
	}
	dims := make([]int64, 0, q.Dims())
	for i := int64(0); i < q.Dims()-2; i++ {
		dims = append(dims, 1)
	}
	dims = append(dims, l, s)
	storage := q.Storage()
	if storage == nil {
		storage = k.Storage()
	}
	return tensor.FromFloat32(storage, mask,
		tensor.WithShapes(dims...),
		tensor.WithDevice(layer.device))
}

func (layer *Attention1) Params() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"w": layer.w,
		"b": layer.b,
	}
}

func (layer *Attention1) Args() map[string]float32 {
	return map[string]float32{
		"dims":    float32(layer.dims),
		"heads":   float32(layer.heads),
		"dropout": float32(layer.dropout),
	}
}

func (layer *Attention1) Freeze() {
	layer.w.SetRequiresGrad(false)
	layer.b.SetRequiresGrad(false)
}

func (layer *Attention1) Unfreeze() {
	layer.w.SetRequiresGrad(true)
	layer.b.SetRequiresGrad(true)
}
