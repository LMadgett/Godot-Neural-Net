using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    internal class Neuron
    {
        public delegate double ActivationFunction(double value);
        private ActivationFunction activationFunction;
        private double bias;
        private double[] weights;

        public Neuron(double bias, double[] weights, ActivationFunction activationFunction)
        {
            this.bias = bias;
            this.weights = weights;
            this.activationFunction = activationFunction;
        }

        public double CalcOutput(double[] inputs)
        {
            double outputValue = 0;
            for(int i = 0; i < inputs.Length; i++)
            {
                outputValue += inputs[i] * weights[i];
            }
            outputValue += bias;
            outputValue = activationFunction(outputValue);
            return outputValue;
        }
    }
}
