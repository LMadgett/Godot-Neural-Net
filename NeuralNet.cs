using Godot;
using System;
using System.IO;
using System.Collections.Generic;

namespace NeuralNet
{
	public partial class NeuralNet : Node
	{
		public int[] layerSizes;
		public List<Layer> layers;
		private Layer inputLayer;
		private Layer outputLayer;

        public bool threadCompleted = false;
        public int currentIteration = 0;
        public int currentInputIndex = 0;
        public double currentError = 1.0;
        public double currentMaxError = 0.0;
        public int currentMaxErrorIndex = 0;

        public delegate double ActivationFunction(double value);

		// Called when the node enters the scene tree for the first time.
		public override void _Ready()
		{

		}

        public void Initialise(int[] layerSizes)
        {
            this.layerSizes = layerSizes;
            InitialiseLayers();
        }

        public void Train(Object trainingParams)
        {
            threadCompleted = false;

            Object[] paramsArray = (Object[])trainingParams;
            double[][] trainingInputs = (double[][])paramsArray[0];
            double[][] expectedOutputs = (double[][])paramsArray[1];
            int numPasses = (int)paramsArray[2];
            double learningRate = (double)paramsArray[3];

            Train(trainingInputs, expectedOutputs, numPasses, learningRate);

            threadCompleted = true;
        }

        public void Train(double[][] trainingInputs, double[][] expectedOutputs, int numPasses, double learningRate)
        {
            for (int pass = 0; pass < numPasses; pass++)
            {
                double totalPassError = 0.0;
                currentIteration = pass;
                currentMaxError = 0.0;

                for (int i = 0; i < trainingInputs.GetLength(0); i++)
                {
                    currentInputIndex = i;
                    double[] outputs = GetOutputs(trainingInputs[i]);

                    BackPropagate(expectedOutputs[i], learningRate);
                    double error = CalculateError(outputs, expectedOutputs[i]);

                    totalPassError += error;
                    if(currentMaxError < error)
                    {
                        currentMaxError = error;
                        currentMaxErrorIndex = i;
                    }
                }
                currentError = totalPassError / trainingInputs.GetLength(0);
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
                outputDeltas[i] = error * SigmoidDerivative(output);
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
                    layerDeltas[i] = sum * SigmoidDerivative(output);
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

        public double SigmoidDerivative(double value)
        {
            return value * (1 - value);
        }

        public double PassThroughActivationFunction(double value)
        {
            return value;
        }

        public void SaveWeights(string filePath, string fileName)
        {
            using (StreamWriter outputFile = new StreamWriter(Path.Combine(filePath, fileName)))
            {
                outputFile.WriteLine("LayerCount: " + layers.Count);
                for(int i = 0; i < layers.Count; i++)
                {
                    outputFile.Write(layerSizes[i]);
                    if(i < layers.Count - 1)
                    {
                        outputFile.Write(",");
                    }
                    else
                    {
                        outputFile.Write("\n");
                    }
                }
                foreach(Layer layer in layers)
                {
                    int layerIdx = layers.IndexOf(layer);
                    double[] biases = layer.biases;
                    double[,] weights = layer.weights; //[layerSize, prevLayerSize]
                    outputFile.WriteLine("Layer[" + layerIdx + "]" + ":Size=" + layerSizes[layerIdx]);

                    if (biases == null)
                    {
                        outputFile.WriteLine("Biases: null");
                    }
                    else
                    {
                        outputFile.WriteLine("Biases: " + string.Join(",", biases));
                    }

                    if(weights == null)
                    {
                        outputFile.WriteLine("Weights: null");
                    }
                    else
                    {
                        for(int i = 0; i < weights.GetLength(0); i++)
                        {
                            outputFile.Write("Weights[" + i + "]: ");
                            for(int j = 0; j < weights.GetLength(1); j++)
                            {
                                outputFile.Write(weights[i, j]);
                                if(j < weights.GetLength(1) - 1)
                                {
                                    outputFile.Write(",");
                                }
                                else
                                {
                                    outputFile.Write("\n");
                                }
                            }
                        }
                    }
                }
            }
        }

        public void ReadWeights(string filePath, string fileName)
        {
            using (StreamReader inputFile = new StreamReader(Path.Combine(filePath, fileName)))
            {
                //read layerCount
                string layerCountStr = inputFile.ReadLine();
                string[] words = layerCountStr.Split(':');
                int layerCount = int.Parse(words[1].Trim());

                //read layer sizes
                string layerSizesStr = inputFile.ReadLine();
                words = layerSizesStr.Split(',');
                layerSizes = new int[layerCount];
                for(int i = 0; i < layerCount; i++)
                {
                    layerSizes[i] = int.Parse(words[i].Trim());
                }

                //construct layers with random weights initially
                InitialiseLayers();

                //read each layer biases and weights
                for(int layerNum = 0; layerNum < layerCount; layerNum++)
                {
                    //ignore this
                    string layerComment = inputFile.ReadLine();

                    //biases
                    string biasesLine = inputFile.ReadLine();
                    string[] sections = biasesLine.Split(':');
                    string[] biasesStr = sections[1].Split(',');

                    if (biasesStr[0].Trim().Equals("null"))
                    {
                        //input layer
                        layers[layerNum].biases = null;
                    }
                    else
                    {
                        double[] biases = new double[biasesStr.Length];
                        for (int i = 0; i < biasesStr.Length; i++)
                        {
                            biases[i] = double.Parse(biasesStr[i].Trim());
                        }
                        layers[layerNum].biases = biases;
                    }

                    //weights
                    for(int weightsNum = 0; weightsNum < layerSizes[layerNum]; weightsNum++)
                    {
                        string weightsLine = inputFile.ReadLine();
                        sections = weightsLine.Split(':');
                        string[] weightsStr = sections[1].Split(',');

                        if (weightsStr[0].Trim().Equals("null"))
                        {
                            //input layer
                            layers[layerNum].weights = null;
                            break;
                        }
                        else
                        {
                            double[] weights = new double[weightsStr.Length];
                            for (int i = 0; i < weightsStr.Length; i++)
                            {
                                weights[i] = double.Parse(weightsStr[i].Trim());
                            }
                            for (int j = 0; j < weights.Length; j++)
                            {
                                layers[layerNum].weights[weightsNum, j] = weights[j];
                            }
                        }
                    }
                }
            }
        }
	}
}
