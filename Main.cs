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
            ReadData();
            InitialiseNeuralNet();
            TrainNeuralNet();
        }

        private void ReadData()
        {
            ReadMNISTTrainingData();
            //DisplayMNISTImage(0);
        }

        public void DisplayMNISTImage(int idx)
        {
            Image img = ConvertMNISTToImage(trainingImages, idx);
            imageRect.Texture = ImageTexture.CreateFromImage(img);
        }

        private void InitialiseNeuralNet()
        {
            neuralNet.Initialise(new int[] { 784, 128, 10 });
        }

        private void TrainNeuralNet()
        {
            int numEncode = 10;
            int numPasses = 100;
            int totalPasses = 1000;
            double learningRate = 0.1;
            double[][] normImages = EncodeImages(numEncode);
            double[][] normLabels = OneHotEncodeLabels(numEncode);
            for (int i = 0; i < totalPasses; i += numPasses)
            {
                neuralNet.Train(normImages, normLabels, numPasses, learningRate);
            }
        }

        private double[][] OneHotEncodeLabels(int numEncode)
        {
            double[][] normLabels = new double[numEncode][];
            for (int i = 0; i < numEncode; i++)
            {
                double[] oneHotVector = new double[10];
                oneHotVector[trainingLabels[i]] = 1.0; // One-hot encoding
                normLabels[i] = oneHotVector;
            }
            return normLabels;
        }

        private double[][] EncodeImages(int numEncode)
        {
            double[][] normImages = new double[numEncode][];
            for (int i = 0; i < numEncode; i++)
            {
                double[] imageVector = new double[784]; // 28x28 = 784 pixels
                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        imageVector[y * 28 + x] = trainingImages[i, y, x] / 255.0;
                    }
                }
                normImages[i] = imageVector;
            }
            return normImages;
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
