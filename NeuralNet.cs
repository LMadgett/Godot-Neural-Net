using Godot;
using System;
using System.Collections.Generic;

namespace NeuralNet
{
	public partial class NeuralNet : Node
	{
		[Export]
		public int numLayers = 3;
		public List<Layer> layers;

		// Called when the node enters the scene tree for the first time.
		public override void _Ready()
		{
			layers = new List<Layer>();
			int prevLayerSize = 0;
			for(int i = 0; i < numLayers; i++)
			{
				Layer layer = new Layer(3, prevLayerSize, MyActivationFunction);
				layers.Add(layer);

				prevLayerSize = layer.GetLayerSize();
			}
		}

		public double MyActivationFunction(double value)
		{
			return value;
		}

		// Called every frame. 'delta' is the elapsed time since the previous frame.
		public override void _Process(double delta)
		{

		}
	}
}
