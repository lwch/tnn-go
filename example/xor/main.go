package main

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/loss"
	"github.com/lwch/gotorch/tensor"
	"github.com/lwch/runtime"
	"github.com/lwch/tnn/nn/layer"
	"github.com/lwch/tnn/nn/layer/activation"
	"github.com/lwch/tnn/nn/net"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const lr = 1e-4
const hiddenSize = 10
const epoch = 10000
const modelFile = "xor.model"
const device = consts.KCPU

var lossFunc = loss.NewMse

func main() {
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		train()
		return
	}
	m := loadModel()
	nextTrain(m)
	predict(m)
}

func train() {
	hidden := layer.NewLinear("hidden", 2, hiddenSize, layer.WithDevice(device))
	outputLayer := layer.NewLinear("output", hiddenSize, 1, layer.WithDevice(device))

	net := net.New(device)
	net.Add(hidden)
	net.Add(activation.NewReLU())
	net.Add(outputLayer)
	// optimizer := optimizer.NewSGD(lr, 0)

	m := newModel(net)

	p := plot.New()
	p.Title.Text = "xor train model"
	p.X.Label.Text = "epoch"
	p.Y.Label.Text = "loss"

	var lossPoints plotter.XYs
	begin := time.Now()
	for i := 0; i < epoch; i++ {
		x, y := getBatch()
		loss := m.Train(x, y)
		if i%10 == 0 {
			acc := accuracy(m)
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n",
				i, loss, acc)
			lossPoints = append(lossPoints, plotter.XY{X: float64(i), Y: float64(loss)})
		}
	}
	fmt.Printf("train cost: %s, param count: %d\n",
		time.Since(begin).String(), net.ParamCount())
	fmt.Println("predict:")
	predict(m)

	lossLine, err := plotter.NewLine(lossPoints)
	runtime.Assert(err)
	lossLine.LineStyle.Color = plotutil.DarkColors[0]

	p.Legend.Add("loss", lossLine)
	p.Legend.XOffs = -20
	p.Legend.YOffs = 5 * vg.Inch

	p.Add(lossLine)
	p.Save(8*vg.Inch, 8*vg.Inch, "xor.png")

	runtime.Assert(net.Save(modelFile))
}

func loadModel() *model {
	net := net.New(device)
	runtime.Assert(net.Load(modelFile))

	return newModel(net)
}

func nextTrain(m *model) {
	for i := 0; i < 1000; i++ {
		x, y := getBatch()
		loss := m.Train(x, y)
		if i%100 == 0 {
			acc := accuracy(m)
			fmt.Printf("Epoch: %d, Loss: %e, Accuracy: %.02f%%\n",
				i, loss, acc)
		}
	}
}

func predict(m *model) {
	x, _ := getBatch()
	xs := x.Float32Value()
	ys := m.Predict(x)
	for i := 0; i < 4; i++ {
		start := i * 2
		fmt.Printf("%d xor %d: %.2f\n",
			int(xs[start]), int(xs[start+1]),
			ys[i])
	}
}

func accuracy(m *model) float32 {
	x, y := getBatch()
	pred := m.Predict(x)
	values := y.Float32Value()
	var correct float32
	for i := 0; i < 4; i++ {
		diff := float32(math.Abs(float64(pred[i]) -
			float64(values[i])))
		if diff < 1 {
			correct += 1 - diff
		}
	}
	return float32(correct) * 100 / 4
}

func getBatch() (*tensor.Tensor, *tensor.Tensor) {
	x := tensor.FromFloat32([]float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}, tensor.WithShapes(4, 2), tensor.WithDevice(device))
	y := tensor.FromFloat32([]float32{
		0,
		1,
		1,
		0,
	}, tensor.WithShapes(4, 1), tensor.WithDevice(device))
	return x, y
}
