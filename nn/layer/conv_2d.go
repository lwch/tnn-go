package layer

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

type Conv2D struct {
	base
	inC, outC int
	kernel    [2]int
	stride    [2]int
	padding   [2]int
	dilation  int
	groups    int
	// params
	w *tensor.Tensor
}

func NewConv2D(name string, inC, outC int, kernel1, kernel2 int, opts ...LayerCreateOption) *Conv2D {
	var layer Conv2D
	layer.new("conv2d", name, opts...)
	layer.inC = inC
	layer.outC = outC
	layer.kernel = [2]int{kernel1, kernel2}
	layer.stride = [2]int{1, 1}
	layer.padding = [2]int{0, 0}
	layer.dilation = 1
	layer.groups = 1
	layer.w = layer.initW(int64(outC), int64(inC), int64(kernel1), int64(kernel2))
	return &layer
}

func (layer *Conv2D) SetStride(stride1, stride2 int) {
	layer.stride = [2]int{stride1, stride2}
}

func (layer *Conv2D) SetPadding(padding1, padding2 int) {
	layer.padding = [2]int{padding1, padding2}
}

func (layer *Conv2D) SetDilation(dilation int) {
	layer.dilation = dilation
}

func (layer *Conv2D) SetGroups(groups int) {
	layer.groups = groups
}

func LoadConv2D(name string, params []*tensor.Tensor, args map[string]float32) Layer {
	var layer Conv2D
	layer.new("conv2d", name)
	layer.inC = int(args["inC"])
	layer.outC = int(args["outC"])
	layer.kernel = [2]int{int(args["kernel1"]), int(args["kernel2"])}
	layer.stride = [2]int{int(args["stride1"]), int(args["stride2"])}
	layer.padding = [2]int{int(args["padding1"]), int(args["padding2"])}
	layer.dilation = int(args["dilation"])
	layer.groups = int(args["groups"])
	layer.w = params[0]
	return &layer
}

func (layer *Conv2D) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.Conv2D(layer.w, nil,
		tensor.Conv2DStride(layer.stride[0], layer.stride[1]),
		tensor.Conv2DPadding(layer.padding[0], layer.padding[1]),
		tensor.Conv2DDilation(layer.dilation),
		tensor.Conv2DGroups(layer.groups))
}

func (layer *Conv2D) Params() []*tensor.Tensor {
	return []*tensor.Tensor{
		layer.w,
	}
}

func (layer *Conv2D) Args() map[string]float32 {
	return map[string]float32{
		"inC":      float32(layer.inC),
		"outC":     float32(layer.outC),
		"kernel1":  float32(layer.kernel[0]),
		"kernel2":  float32(layer.kernel[1]),
		"stride1":  float32(layer.stride[0]),
		"stride2":  float32(layer.stride[1]),
		"padding1": float32(layer.padding[0]),
		"padding2": float32(layer.padding[1]),
		"dilation": float32(layer.dilation),
		"groups":   float32(layer.groups),
	}
}

func (layer *Conv2D) Freeze() {
	layer.w.SetRequiresGrad(false)
}

func (layer *Conv2D) Unfreeze() {
	layer.w.SetRequiresGrad(true)
}

func (layer *Conv2D) ToScalarType(t consts.ScalarType) {
	layer.w = layer.w.ToScalarType(t)
}

func (layer *Conv2D) Reset() {
	layer.w = layer.initW(layer.w.Shapes()...)
}
