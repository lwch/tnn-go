package layer

import (
	"github.com/lwch/tnn/internal/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Rnn struct {
	*base
	featureSize, steps int
	hidden             int
	// params
	w *gorgonia.Node
	b *gorgonia.Node
}

func NewRnn(featureSize, steps, hidden int) *Rnn {
	var layer Rnn
	layer.base = new("rnn")
	layer.featureSize = featureSize
	layer.steps = steps
	layer.hidden = hidden
	return &layer
}

func LoadRnn(g *gorgonia.ExprGraph, name string, params map[string]*pb.Dense, args map[string]float32) Layer {
	var layer Rnn
	layer.base = new("rnn")
	layer.name = name
	layer.featureSize = int(args["feature_size"])
	layer.steps = int(args["steps"])
	layer.hidden = int(args["hidden"])
	layer.w = loadParam(g, params["w"], "w")
	layer.b = loadParam(g, params["b"], "b")
	return &layer
}

func buildRnnBlock(x *gorgonia.Node, nodes []*gorgonia.Node, names []string, batchSize, featureSize, steps, hidden int) []*gorgonia.Node {
	if nodes[0] == nil {
		nodes[0] = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(featureSize, hidden),
			gorgonia.WithName(names[0]),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if nodes[1] == nil {
		nodes[1] = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(hidden, hidden),
			gorgonia.WithName(names[1]),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if nodes[2] == nil {
		nodes[2] = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(batchSize, hidden),
			gorgonia.WithName(names[2]),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	if nodes[3] == nil {
		nodes[3] = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(batchSize, hidden),
			gorgonia.WithName(names[3]),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	return nodes
}

func (layer *Rnn) Forward(x, h *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
	inputShape := x.Shape()
	if layer.w == nil {
		layer.w = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(layer.hidden, layer.featureSize+layer.hidden),
			gorgonia.WithName("w"),
			gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	}
	if layer.b == nil {
		layer.b = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(inputShape[0], layer.hidden),
			gorgonia.WithName("b"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	if h == nil {
		h = gorgonia.NewMatrix(x.Graph(), gorgonia.Float32,
			gorgonia.WithShape(inputShape[0], layer.hidden), gorgonia.WithName("h"),
			gorgonia.WithInit(gorgonia.Zeroes()))
	}
	x, err := gorgonia.Transpose(x, 1, 0, 2) // (steps, batch, feature)
	if err != nil {
		return nil, nil, err
	}
	var result *gorgonia.Node
	for i := 0; i < layer.steps; i++ {
		t, err := gorgonia.Slice(x, gorgonia.S(i)) // (batch, feature)
		if err != nil {
			return nil, nil, err
		}
		z := gorgonia.Must(gorgonia.Concat(1, h, t))           // (batch, feature+hidden)
		wt := gorgonia.Must(gorgonia.Transpose(layer.w, 1, 0)) // (feature+hidden, hidden)
		z = gorgonia.Must(gorgonia.Mul(z, wt))                 // (batch, hidden)
		z = gorgonia.Must(gorgonia.Add(z, layer.b))            // (batch, hidden)
		h = gorgonia.Must(gorgonia.Tanh(z))                    // (batch, hidden)
		if result == nil {
			result = h
		} else {
			result = gorgonia.Must(gorgonia.Concat(0, result, h))
		}
	}
	return gorgonia.Must(gorgonia.Reshape(result,
		tensor.Shape{inputShape[0], inputShape[1], layer.hidden})), h, nil
}

func (layer *Rnn) Params() gorgonia.Nodes {
	return gorgonia.Nodes{layer.w, layer.b}
}

func (layer *Rnn) Args() map[string]float32 {
	return map[string]float32{
		"feature_size": float32(layer.featureSize),
		"steps":        float32(layer.steps),
		"hidden":       float32(layer.hidden),
	}
}
