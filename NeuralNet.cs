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
		public List<Layer> layers;

		private Layer inputLayer;
		private Layer outputLayer;

		public delegate double ActivationFunction(double value);

		// Called when the node enters the scene tree for the first time.
		public override void _Ready()
		{
			InitialiseLayers();
			double[] outputs = GetOutputs(inputs);
			for(int i = 0; i < outputs.Length; i++)
			{
				GD.Print("outputs[" + i + "] = " + outputs[i]);
			}
		}

		private void InitialiseLayers()
		{
			int numLayers = layerSizes.Length;
            layers = new List<Layer>();
            int prevLayerSize = 0;
            for (int i = 0; i < numLayers; i++)
            {
                Layer layer = new Layer(layerSizes[i], prevLayerSize, StepActivationFunction);
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

		public double StepActivationFunction(double value)
		{
			double output = 0;
			if(value > 0) output = 1;
			return output;
		}

		// Called every frame. 'delta' is the elapsed time since the previous frame.
		public override void _Process(double delta)
		{

		}
	}
}
