package layer

import (
	"github.com/lwch/tnn/internal/math"
	"github.com/lwch/tnn/internal/pb"
	"github.com/lwch/tnn/nn/initializer"
	"github.com/lwch/tnn/nn/tensor"
	"gonum.org/v1/gonum/mat"
)

type Lstm struct {
	*base
	featureSize, steps int
	hidden             int
}

func NewLstm(featureSize, steps, hidden int, init initializer.Initializer) Layer {
	var layer Lstm
	layer.base = new("lstm", map[string]Shape{
		// It
		"Wii": {featureSize, hidden},
		"Bii": {1, hidden},
		"Whi": {hidden, hidden},
		"Bhi": {1, hidden},
		// Ft
		"Wif": {featureSize, hidden},
		"Bif": {1, hidden},
		"Whf": {hidden, hidden},
		"Bhf": {1, hidden},
		// Gt
		"Wig": {featureSize, hidden},
		"Big": {1, hidden},
		"Whg": {hidden, hidden},
		"Bhg": {1, hidden},
		// Ot
		"Wio": {featureSize, hidden},
		"Bio": {1, hidden},
		"Who": {hidden, hidden},
		"Bho": {1, hidden},
	}, init)
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadLstm(name string, params map[string]*pb.Dense, args map[string]*pb.Dense) Layer {
	arr := args["params"].GetData()
	var layer Lstm
	layer.featureSize = int(arr[0])
	layer.steps = int(arr[1])
	layer.hidden = int(arr[2])
	layer.base = new("lstm", nil, nil)
	layer.name = name
	layer.base.loadParams(params)
	return &layer
}

// Forward https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
func (layer *Lstm) Forward(input *tensor.Tensor, isTraining bool) *tensor.Tensor {
	if !layer.hasInit {
		layer.initParams()
	}
	// It
	Wii := layer.params.Get("Wii")
	Bii := layer.params.Get("Bii")
	Whi := layer.params.Get("Whi")
	Bhi := layer.params.Get("Bhi")
	// Ft
	Wif := layer.params.Get("Wif")
	Bif := layer.params.Get("Bif")
	Whf := layer.params.Get("Whf")
	Bhf := layer.params.Get("Bhf")
	// Gt
	Wig := layer.params.Get("Wig")
	Big := layer.params.Get("Big")
	Whg := layer.params.Get("Whg")
	Bhg := layer.params.Get("Bhg")
	// Ot
	Wio := layer.params.Get("Wio")
	Bio := layer.params.Get("Bio")
	Who := layer.params.Get("Who")
	Bho := layer.params.Get("Bho")
	batchSize, _ := input.Dims()
	c := tensor.New(nil, batchSize, layer.hidden)
	h := tensor.New(nil, batchSize, layer.hidden)
	for t := layer.steps - 1; t >= 0; t-- {
		start := t * layer.featureSize
		x := input.Slice(0, batchSize, start, start+layer.featureSize)
		It := math.Sigmoid(x.Mul(Wii).AddVector(Bii).Add(h.Mul(Whi)).AddVector(Bhi))
		Ft := math.Sigmoid(x.Mul(Wif).AddVector(Bif).Add(h.Mul(Whf)).AddVector(Bhf))
		Gt := x.Mul(Wig).AddVector(Big).Add(h.Mul(Whg)).AddVector(Bhg).Tanh()
		Ot := math.Sigmoid(x.Mul(Wio).AddVector(Bio).Add(h.Mul(Who)).AddVector(Bho))
		c = Ft.MulElem(c).Add(It.MulElem(Gt))
		h = Ot.MulElem(c.Tanh())
	}
	return h
}

func (layer *Lstm) Args() map[string]*mat.VecDense {
	return map[string]*mat.VecDense{
		"params": mat.NewVecDense(3, []float64{
			float64(layer.featureSize),
			float64(layer.steps),
			float64(layer.hidden)}),
	}
}