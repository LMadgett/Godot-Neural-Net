using Godot;
using System;
using System.Collections;
using System.Threading;

namespace NeuralNet
{
    public partial class Main : Node2D
    {
        [Export]
        NeuralNet neuralNet;
        [Export]
        TextureRect trainRect;
        [Export]
        TextureRect maxErrorRect;
        [Export]
        Label progressLabel;

        byte[,,] trainingImages;
        byte[] trainingLabels;

        double[][] normImages;
        double[][] normLabels;

        private bool trained = false;
        private bool training = false;
        private Thread trainThread;

        int currentIteration = 0;
        int maxIterations = 10;
        int taskIterSize = 100;
        int numImagesToEncode = 100;
        double learningRate = 0.1;

        private Hashtable mnistTextures = new Hashtable();

        // Called when the node enters the scene tree for the first time.
        public override void _Ready()
        {
            ReadData();
            InitialiseNeuralNet();
        }

        private void ReadData()
        {
            ReadMNISTTrainingData();
            //DisplayMNISTImage(0);
        }

        public void DisplayMNISTImage(int idx, TextureRect rect)
        {
            Texture2D tex = null;
            if(mnistTextures.ContainsKey(idx))
            {
                tex = (Texture2D)mnistTextures[idx];
            }
            else
            {
                Image img = ConvertMNISTToImage(trainingImages, idx);
                tex = ImageTexture.CreateFromImage(img);
                mnistTextures[idx] = tex;
            }
            rect.Texture = tex;
        }

        private void InitialiseNeuralNet()
        {
            neuralNet.Initialise(new int[] { 784, 128, 64, 10 });
        }

        private void TrainNeuralNet()
        {
            training = true;
            if(normImages == null)
            {
                normImages = EncodeImages(numImagesToEncode);
            }
            if(normLabels == null)
            {
                normLabels = OneHotEncodeLabels(numImagesToEncode);
            }

            //no thread training?
            if (trainThread == null && currentIteration < maxIterations)
            {
                trainThread = new Thread(new ParameterizedThreadStart(neuralNet.Train));
                Object trainingParams = new Object[] { normImages, normLabels, taskIterSize, learningRate };
                trainThread.Start(trainingParams);
                currentIteration++;

                GD.Print($"Training iteration {currentIteration}/{maxIterations}.");
            }
            else
            {
                if (trainThread != null)
                {
                    if (trainThread.ThreadState == ThreadState.Stopped || trainThread.ThreadState == ThreadState.Unstarted)
                    {
                        training = false;
                        trainThread = null;

                        if (currentIteration >= maxIterations)
                        {
                            trained = true;

                            GD.Print("Training completed.");
                            progressLabel.Text = progressLabel.Text + "\n\n Training completed.";
                        }
                    }
                    else
                    {
                        int currentThreadIteration = neuralNet.currentIteration;
                        int currentThreadInputIndex = neuralNet.currentInputIndex;
                        double currentThreadError = neuralNet.currentError;
                        double currentThreadMaxError = neuralNet.currentMaxError;
                        int currentThreadMaxErrorIndex = neuralNet.currentMaxErrorIndex;

                        progressLabel.Text = $"NumImagesToTrain: {numImagesToEncode}\nTotalIterations: {maxIterations * taskIterSize}\n\nTraining: Iteration {currentIteration}/{maxIterations}\nThreadIteration: {currentThreadIteration + 1}/{taskIterSize}\nSample: {currentThreadInputIndex + 1}/{normImages.Length}\nAvgError: {currentThreadError:F10}\nMaxError: {currentThreadMaxError:F10}";

                        DisplayMNISTImage(currentThreadInputIndex, trainRect);
                        DisplayMNISTImage(currentThreadMaxErrorIndex, maxErrorRect);
                    }
                }
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
            if(!trained)
            {
                TrainNeuralNet();
            }
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
