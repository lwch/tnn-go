package layer

import (
	"math"

	"github.com/lwch/tnn/nn/vector"
	"gonum.org/v1/gonum/mat"
)

type MaxPool struct {
	*base
	padedShape              Shape
	imageShape, kernelShape Shape
	stride                  Stride
	idx                     [][]int
}

func NewMaxPool(imgShape, kernel Shape, stride Stride) *MaxPool {
	var layer MaxPool
	layer.base = new("conv2d", nil, nil, layer.forward, layer.backward)
	layer.imageShape = imgShape
	layer.kernelShape = kernel
	layer.stride = stride
	return &layer
}

func (layer *MaxPool) OutputShape() Shape {
	dy := float64(layer.imageShape.M - layer.kernelShape.M)
	dx := float64(layer.imageShape.N - layer.kernelShape.N)
	y := math.Ceil(dy/float64(layer.stride.Y)) + 1
	x := math.Ceil(dx/float64(layer.stride.X)) + 1
	return Shape{int(y), int(x)}
}

func (layer *MaxPool) forward(input mat.Matrix) mat.Matrix {
	batch, _ := input.Dims()
	outputShape := layer.OutputShape()
	if !layer.hasInit {
		layer.initParams()
		layer.idx = make([][]int, batch)
		for i := 0; i < batch; i++ {
			layer.idx[i] = make([]int, outputShape.M*outputShape.N)
		}
	}
	pad := layer.pad(input)
	layer.padedShape.M, layer.padedShape.N = pad.Dims()
	maxFunc := func(m mat.Matrix) (float64, int) {
		rows, cols := m.Dims()
		max := math.Inf(-1)
		idx := -1
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				v := m.At(i, j)
				if v > max {
					max = v
					idx = i*cols + j
				}
			}
		}
		return max, idx
	}
	ret := mat.NewDense(batch, outputShape.M*outputShape.N, nil)
	for i := 0; i < pad.Size(); i++ {
		img := pad.Get(i)
		idx := 0
		for j := 0; j < outputShape.M; j++ {
			topLeftY := j * layer.stride.Y
			bottomRightY := topLeftY + layer.kernelShape.M
			for k := 0; k < outputShape.N; k++ {
				topLeftX := k * layer.stride.X
				bottomRightX := topLeftX + layer.kernelShape.N
				rect := img.(vector.Slicer).Slice(topLeftY, bottomRightY, topLeftX, bottomRightX)
				var value float64
				value, layer.idx[i][idx] = maxFunc(rect)
				ret.Set(i, idx, value)
				idx++
			}
		}
	}
	return ret
}

func (layer *MaxPool) backward(grad mat.Matrix) mat.Matrix {
	batch, _ := grad.Dims()
	ret := vector.NewVector3D(batch, layer.padedShape.M, layer.padedShape.N)
	outputShape := layer.OutputShape()
	for i := 0; i < batch; i++ {
		img := ret.Get(i)
		idx := 0
		rv := grad.(vector.RowViewer).RowView(i)
		for j := 0; j < outputShape.M; j++ {
			startY := j * layer.stride.Y
			for k := 0; k < outputShape.N; k++ {
				startX := k * layer.stride.X
				g := rv.AtVec(idx)
				n := layer.idx[i][idx]
				dy := math.Floor(float64(n) / float64(layer.kernelShape.N))
				dx := n % layer.kernelShape.N
				g += img.At(startY+int(dy), startX+dx)
				img.(vector.Seter).Set(startY+int(dy), startX+dx, g)
				idx++
			}
		}
	}
	return ret.Cut(layer.imageShape.M, layer.imageShape.N).ToMatrix()
}

func (layer *MaxPool) pad(input mat.Matrix) *vector.Vector3D {
	padY := layer.stride.Y - (layer.imageShape.M % layer.stride.Y)
	padX := layer.stride.X - (layer.imageShape.N % layer.stride.X)
	reshape := vector.ReshapeMatrix(input, layer.imageShape.M, layer.imageShape.N)
	if padY != layer.stride.Y || padX != layer.stride.X {
		reshape.Pad(padY, padX)
	}
	return reshape
}
