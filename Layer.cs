using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Layer
    {
        public Neuron[] neurons;
        private double[] outputs;
        int prevLayerSize;
        Neuron.ActivationFunction activationFunction;
        Random rand = new Random(27);

        public Layer(int layerSize, int prevLayerSize, Neuron.ActivationFunction activation)
        {
            neurons = new Neuron[layerSize];
            this.prevLayerSize = prevLayerSize;
            this.activationFunction = activation;
            outputs = new double[layerSize];
            InitialiseNeurons();
        }

        private void InitialiseNeurons()
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                double[] weights = new double[prevLayerSize];
                for(int j = 0; j < weights.Length; j++)
                {
                    weights[j] = rand.NextDouble() * 2 - 1;
                }
                double bias = rand.NextDouble() * 2 - 1;
                neurons[i] = new Neuron(bias, weights, activationFunction);
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
            return neurons.Length;
        }

        public void CalcOutputs(Layer prevLayer)
        {
            for(int i = 0; i < neurons.Length; i++)
            {
                Neuron neuron = neurons[i];
                double[] inputs = prevLayer.outputs;
                double output = neuron.CalcOutput(inputs);
                outputs[i] = output;
            }
        }
    }
}
