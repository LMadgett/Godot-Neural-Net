using Godot;
using System;
using System.Collections.Generic;

namespace NeuralNet
{
	public partial class NeuralNet : Node
	{
		[Export]
		public int[] layerSizes;
		[Export]
		public double[] inputs;
		[Export]
		public double[] expectedOutputs;
        [Export]
        public int numPasses = 10000;
        [Export]
        public double learningRate = 0.1;
        [Export]
        public int printInterval = 1000;

		public List<Layer> layers;
		private Layer inputLayer;
		private Layer outputLayer;

		public delegate double ActivationFunction(double value);

		// Called when the node enters the scene tree for the first time.
		public override void _Ready()
		{
			InitialiseLayers();

            for(int pass = 0; pass < numPasses; pass++)
            {
                double[] outputs = GetOutputs(inputs);
                if (pass % printInterval == 0)
                {
                    for (int i = 0; i < outputs.Length; i++)
                    {
                        GD.Print("outputs[" + i + "] = " + outputs[i]);
                    }
                    double error = CalculateError(outputs, expectedOutputs);
                    GD.Print("Error: " + error);
                }
                BackPropagate(expectedOutputs, learningRate);
            }
		}

        public void BackPropagate(double[] expectedOutputs, double learningRate = 0.1)
        {
            if (expectedOutputs.Length != outputLayer.GetLayerSize())
            {
                throw new ArgumentException("NeuralNet.BackPropagate() expectedOutputs.Length != outputLayer.GetLayerSize()");
            }

            // Store deltas for each layer
            var deltas = new List<double[]>(layers.Count);

            // 1. Output layer delta
            var outputDeltas = new double[outputLayer.layerSize];
            for (int i = 0; i < outputLayer.layerSize; i++)
            {
                double output = outputLayer.GetLayerOutputs()[i];
                double error = expectedOutputs[i] - output;
                // Sigmoid derivative: output * (1 - output)
                outputDeltas[i] = error * output * (1 - output);
            }
            deltas.Insert(0, outputDeltas);

            // 2. Hidden layers deltas (backwards)
            for (int l = layers.Count - 2; l > 0; l--)
            {
                var layer = layers[l];
                var nextLayer = layers[l + 1];
                var layerDeltas = new double[layer.layerSize];
                double[] layerOutputs = layer.GetLayerOutputs();
                for (int i = 0; i < layer.layerSize; i++)
                {
                    double output = layerOutputs[i];
                    double sum = 0.0;
                    for (int j = 0; j < nextLayer.layerSize; j++)
                    {
                        sum += nextLayer.weights[j, i] * deltas[0][j];
                    }
                    layerDeltas[i] = sum * output * (1 - output);
                }
                deltas.Insert(0, layerDeltas);
            }

            // 3. Update weights and biases
            for (int l = 1; l < layers.Count; l++)
            {
                var layer = layers[l];
                var prevOutputs = layers[l - 1].GetLayerOutputs();
                var layerDeltas = deltas[l - 1];

                // Update weights
                for (int i = 0; i < layer.layerSize; i++)
                {
                    for (int j = 0; j < layer.weights.GetLength(1); j++)
                    {
                        layer.weights[i, j] += learningRate * layerDeltas[i] * prevOutputs[j];
                    }
                    // Update bias
                    layer.biases[i] += learningRate * layerDeltas[i];
                }
            }
        }

        private void InitialiseLayers()
		{
			int numLayers = layerSizes.Length;
            layers = new List<Layer>();
            int prevLayerSize = 0;
            for (int i = 0; i < numLayers; i++)
            {
                Layer layer = new Layer(layerSizes[i], prevLayerSize, SigmoidActivationFunction);
                layers.Add(layer);

                prevLayerSize = layer.GetLayerSize();
            }

            inputLayer = layers[0];
            outputLayer = layers[layers.Count - 1];
        }

		public double[] GetOutputs(double[] inputs)
		{
			inputLayer.SetOutputsFromInputs(inputs);
			for (int i = 1; i < layers.Count; i++)
			{
                Layer prevLayer = layers[i - 1];
                Layer currentLayer = layers[i];
                currentLayer.CalcOutputs(prevLayer);
            }

			double[] outputs = outputLayer.GetLayerOutputs();
			return outputs;
		}

		private double CalculateError(double[] outputs, double[] expectedOutputs)
		{
            if(outputs.Length != expectedOutputs.Length)
			{
                GD.PrintErr("Outputs and expected outputs must have the same length.");
                return 0;
            }
			
            double error = 0;
            for(int i = 0; i < outputs.Length; i++)
			{
                double diff = outputs[i] - expectedOutputs[i];
                error += diff * diff; // Squared error
            }
            return error / outputs.Length; // Mean squared error
        }

		public double StepActivationFunction(double value)
		{
			double output = 0;
			if(value > 0) output = 1;
			return output;
		}

		public double SigmoidActivationFunction(double value)
		{
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        public double PassThroughActivationFunction(double value)
        {
            return value;
        }

        // Called every frame. 'delta' is the elapsed time since the previous frame.
        public override void _Process(double delta)
		{

		}
	}
}
