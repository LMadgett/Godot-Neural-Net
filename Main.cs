using Godot;
using System;
using System.IO;
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
        [Export]
        Button trainButton;
        [Export]
        Button saveWeightsButton;
        [Export]
        Button readWeightsButton;
        [Export]
        FileDialog fileDialog;
        [Export]
        LineEdit fileLineEdit;

        byte[,,] trainingImages;
        byte[] trainingLabels;

        double[][] normImages;
        double[][] normLabels;

        private bool trained = false;
        private bool training = false;
        private bool doTraining = false;
        private Thread trainThread;

        private bool weightsSaved = false;

        int currentIteration = 0;
        int maxIterations = 10;
        int taskIterSize = 100;
        int numImagesToEncode = 10;
        double learningRate = 0.1;

        private string filePath;
        private string fileName;

        private Hashtable mnistTextures = new Hashtable();

        private long trainingStartTime = 0;

        // Called when the node enters the scene tree for the first time.
        public override void _Ready()
        {
            ReadData();
            InitialiseNeuralNet();
        }

        private long GetCurrentTimeMillis()
        {
            DateTimeOffset date = DateTimeOffset.UtcNow;
            return date.ToUnixTimeMilliseconds();
        }

        public void ReadWeights()
        {
            if (fileName != null && fileName.Length > 0)
            {
                neuralNet.ReadWeights(filePath, fileName);
                GD.Print("Weights read from " + fileName);
            }
        }

        public void SaveWeights()
        {
            if (fileName != null && fileName.Length > 0)
            {
                neuralNet.SaveWeights(filePath, fileName);
                weightsSaved = true;
                GD.Print("Weights saved to " + fileName);
            }
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
                        
                        int totalIterations = maxIterations * taskIterSize;
                        int currentIter = ((currentIteration - 1) * taskIterSize) + currentThreadIteration;

                        long timeSoFar = GetCurrentTimeMillis() - trainingStartTime;
                        double timePerIteration = ((double)timeSoFar / (double)currentIter);
                        double estimatedTotalTime = (timePerIteration * totalIterations) / 1000;
                        double timeLeft = (totalIterations - currentIter) * timePerIteration / 1000.0;
                        double percentageComplete = ((double)currentIter / (double)totalIterations) * 100.0;

                        progressLabel.Text = $"NumImagesToTrain: {numImagesToEncode}\nTotalIterations: {maxIterations * taskIterSize}\n\nTraining: Iteration {currentIteration}/{maxIterations}\nThreadIteration: {currentThreadIteration + 1}/{taskIterSize}\nSample: {currentThreadInputIndex + 1}/{normImages.Length}\nAvgError: {currentThreadError:F10}\nMaxError: {currentThreadMaxError:F10}\nPercentComplete: {percentageComplete:F3}\nEstTotalTime (s): {estimatedTotalTime:F3}\nTimeLeft (s): {timeLeft:F3}";

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

        // Called every frame. 'delta' is the elapsed time since the previous frame.
        public override void _Process(double delta)
        {
            if (!trained && doTraining)
            {
                TrainNeuralNet();
            }
        }

        private void OnTrainButtonPressed()
        {
            if(doTraining)
            {
                doTraining = false;
                trainButton.Text = "Start Training";
                readWeightsButton.Disabled = false;
                saveWeightsButton.Disabled = false;
            }
            else
            {
                doTraining = true;
                trainButton.Text = "Stop Training";
                readWeightsButton.Disabled = true;
                saveWeightsButton.Disabled = true;
                trainingStartTime = GetCurrentTimeMillis();
            }
        }

        private void OnLoadButtonPressed()
        {
            ReadWeights();
        }

        private void OnSaveButtonPressed()
        {
            SaveWeights();
        }

        private void OnPickButtonPressed()
        {
            fileDialog.Visible = true;
        }

        private void OnFileDialogFileSelected(string filePicked)
        {
            fileLineEdit.Text = filePicked;
            filePath = Path.GetDirectoryName(filePicked);
            fileName = Path.GetFileName(filePicked);
        }

        private void OnFileLineEditTextSubmitted(string fileLine)
        {
            filePath = Path.GetDirectoryName(fileLine);
            fileName = Path.GetFileName(fileLine);
        }
    }
}
