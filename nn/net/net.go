package net

import (
	"fmt"
	"tnn/nn/layer"
	"tnn/nn/layer/activation"
	"tnn/nn/params"
	"tnn/nn/pb"

	"gonum.org/v1/gonum/mat"
)

type loadFunc func(string, map[string]*pb.Dense) layer.Layer

var loadFuncs = map[string]loadFunc{
	"dense":   layer.LoadDense,
	"dropout": layer.LoadDropout,
	// activation
	"sigmoid":  activation.Load("sigmoid"),
	"softplus": activation.Load("softplus"),
	"tanh":     activation.Load("tanh"),
	"relu":     activation.Load("relu"),
}

type Net struct {
	layers []layer.Layer
}

func New() *Net {
	return &Net{}
}

func (n *Net) Set(layer ...layer.Layer) {
	n.layers = layer
}

func (n *Net) Add(layer layer.Layer) {
	n.layers = append(n.layers, layer)
}

func (n *Net) Forward(input mat.Matrix) mat.Matrix {
	for i := 0; i < len(n.layers); i++ {
		input = n.layers[i].Forward(input)
	}
	return input
}

func (n *Net) Backward(grad mat.Matrix) []*params.Params {
	ret := make([]*params.Params, len(n.layers))
	for i := len(n.layers) - 1; i >= 0; i-- {
		grad = n.layers[i].Backward(grad)
		var p params.Params
		p.Copy(n.layers[i].Context())
		ret[i] = &p
	}
	return ret
}

func (n *Net) Params() []*params.Params {
	ret := make([]*params.Params, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = n.layers[i].Params()
	}
	return ret
}

func (n *Net) ParamCount() uint64 {
	var count uint64
	for i := 0; i < len(n.layers); i++ {
		ps := n.layers[i].Params()
		if ps == nil {
			continue
		}
		ps.Range(func(_ string, val mat.Matrix) {
			rows, cols := val.Dims()
			count += uint64(rows * cols)
		})
	}
	return count
}

func (n *Net) SaveLayers() []*pb.Layer {
	ret := make([]*pb.Layer, len(n.layers))
	for i := 0; i < len(n.layers); i++ {
		ret[i] = new(pb.Layer)
		ret[i].Class = n.layers[i].Class()
		ret[i].Name = n.layers[i].Name()
		ps := n.layers[i].Params()
		if ps == nil {
			continue
		}
		if ps.Size() == 0 {
			continue
		}
		ret[i].Params = make(map[string]*pb.Dense)
		ps.Range(func(key string, val mat.Matrix) {
			var dense pb.Dense
			rows, cols := val.Dims()
			dense.Rows, dense.Cols = int32(rows), int32(cols)
			dense.Data = mat.DenseCopyOf(val).RawMatrix().Data
			ret[i].Params[key] = &dense
		})
	}
	return ret
}

func (n *Net) LoadLayers(layers []*pb.Layer) {
	n.layers = make([]layer.Layer, len(layers))
	for i := 0; i < len(layers); i++ {
		class := layers[i].GetClass()
		fn := loadFuncs[class]
		if fn == nil {
			panic("unsupported " + class + " layer")
		}
		name := layers[i].GetName()
		n.layers[i] = fn(name, layers[i].GetParams())
	}
}

func (n *Net) Print() {
	fmt.Println("Layers:")
	for i := 0; i < len(n.layers); i++ {
		n.layers[i].Print()
	}
}