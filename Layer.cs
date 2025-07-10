using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Layer
    {
        public double[] biases;
        public double[,] weights;
        private double[] outputs;
        int prevLayerSize;
        public int layerSize;
        NeuralNet.ActivationFunction activationFunction;
        Random rand = new Random(27);

        public Layer(int layerSize, int prevLayerSize, NeuralNet.ActivationFunction activation)
        {
            this.layerSize = layerSize;
            this.prevLayerSize = prevLayerSize;
            this.activationFunction = activation;

            outputs = new double[layerSize];
            biases = new double[layerSize];
            weights = new double[layerSize, prevLayerSize];

            InitialiseNeurons();
        }

        private void InitialiseNeurons()
        {
            for (int i = 0; i < layerSize; i++)
            {
                for(int j = 0; j < prevLayerSize; j++)
                {
                    weights[i, j] = rand.NextDouble() * 2 - 1;
                }
                double bias = rand.NextDouble() * 2 - 1;
                biases[i] = bias;
            }
        }

        public void SetOutputsFromInputs(double[] inputs)
        {
            for(int i = 0; i < inputs.Length; i++)
            {
                outputs[i] = inputs[i];
            }
        }

        public double[] GetLayerOutputs()
        {
            return outputs;
        }

        public int GetLayerSize()
        {
            return layerSize;
        }

        public void CalcOutputs(Layer prevLayer)
        {
            for(int i = 0; i < layerSize; i++)
            {
                double[] inputs = prevLayer.outputs;
                double output = CalcNeuronOutput(i, inputs);
                outputs[i] = output;
            }
        }

        public double CalcNeuronOutput(int neuronIdx, double[] inputs)
        {
            double outputValue = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                outputValue += inputs[i] * weights[neuronIdx, i];
            }
            outputValue += biases[neuronIdx];
            outputValue = activationFunction(outputValue);
            return outputValue;
        }
    }
}
