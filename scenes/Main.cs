using Godot;
using System;

namespace NeuralNet
{
    public partial class Main : Node2D
    {
        [Export]
        NeuralNet neuralNet;
        [Export]
        TextureRect imageRect;

        byte[,,] trainingImages;
        byte[] trainingLabels;

        // Called when the node enters the scene tree for the first time.
        public override void _Ready()
        {
            ReadMNISTTrainingData();
            Image img = ConvertMNISTToImage(trainingImages, 0);
            imageRect.Texture = ImageTexture.CreateFromImage(img);
        }

        // Called every frame. 'delta' is the elapsed time since the previous frame.
        public override void _Process(double delta)
        {

        }

        private void ReadMNISTTrainingData()
        {
            trainingImages = (byte[,,])IDXReader.ReadIDX("MNIST/train-images.idx3-ubyte");
            trainingLabels = (byte[])IDXReader.ReadIDX("MNIST/train-labels.idx1-ubyte");
        }

        private void PrintMNISTTrainingLabels()
        {
            for(int i = 0; i < trainingLabels.Length; i++)
            {
                GD.Print($"TrainingLabel {i} = {trainingLabels[i]}");
            }
        }

        private Image ConvertMNISTToImage(byte[,,] imageData, int idx)
        {
            int width = imageData.GetLength(1);
            int height = imageData.GetLength(2);
            Image img = Image.CreateEmpty(width, height, false, Image.Format.Rgb8);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte pixelValue = imageData[idx, y, x]; 
                    Color colour = new Color(pixelValue / 255.0f, pixelValue / 255.0f, pixelValue / 255.0f);
                    img.SetPixel(x, y, colour);
                }
            }

            return img;
        }
    }
}
