using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    internal class Layer
    {
        public Neuron[] neurons;
        private double[] outputs;
        int prevLayerSize;
        Neuron.ActivationFunction activationFunction;

        public Layer(int layerSize, int prevLayerSize, Neuron.ActivationFunction activation)
        {
            neurons = new Neuron[layerSize];
            this.prevLayerSize = prevLayerSize;
            this.activationFunction = activation;
            InitialiseNeurons();
        }

        private void InitialiseNeurons()
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                double[] weights = new double[prevLayerSize];
                double bias = 0.0;
                neurons[i] = new Neuron(bias, weights, activationFunction);
            }
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
