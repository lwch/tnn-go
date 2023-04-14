package layer

import (
	"fmt"
	"tnn/initializer"
	"tnn/nn/pb"
	"tnn/nn/vector"

	"gonum.org/v1/gonum/mat"
)

type Dense struct {
	*base
}

func NewDense(output int, init initializer.Initializer) *Dense {
	var layer Dense
	layer.base = new("dense", map[string]Shape{
		"w": {NoneShape, output}, // rows reshape from input
		"b": {NoneShape, output}, // rows reshape from input
	}, init, layer.forward, layer.backward)
	return &layer
}

func LoadDense(name string, params map[string]*pb.Dense) Layer {
	var layer Dense
	layer.base = new("dense", nil, nil, layer.forward, layer.backward)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

func (layer *Dense) forward(input mat.Matrix) mat.Matrix {
	if !layer.hasInit {
		shapeW := layer.shapes["w"]
		shapeB := layer.shapes["b"]
		_, shapeW.M = input.Dims()
		shapeB.M, _ = input.Dims()
		layer.shapes["w"] = shapeW
		layer.shapes["b"] = shapeB
		layer.initParams()
	}
	var ret mat.Dense
	ret.Mul(input, layer.params["w"])
	ret.Add(&ret, layer.params["b"])
	return &ret
}

func (layer *Dense) backward(grad mat.Matrix) mat.Matrix {
	dw := layer.context["w"]
	db := layer.context["b"]

	dw.(vector.Muler).Mul(layer.input.T(), grad)
	db.(vector.Copyer).Copy(grad)

	var ret mat.Dense
	w := layer.params["w"]
	ret.Mul(grad, w.T())
	return &ret
}

func (layer *Dense) Print() {
	layer.base.Print()
	_, cnt := layer.params["w"].Dims()
	fmt.Println("    Output Count:", cnt)
	fmt.Println("    Params:")
	for name, dense := range layer.params {
		rows, cols := dense.Dims()
		fmt.Println("      - "+name+":", fmt.Sprintf("%dx%d", rows, cols))
	}
}