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
        [Export]
        TextureRect testRect;
        [Export]
        LineEdit TestLineEdit;
        [Export]
        Label testLabel;
        [Export]
        Button testButton;
        [Export]
        Button testAllButton;
        [Export]
        Label numCorrectLabel;

        byte[,,] trainingImages;
        byte[] trainingLabels;

        byte[,,] testImages;
        byte[] testLabels;

        double[][] normTrainingImages;
        double[][] normTrainingLabels;

        double[][] normTestImages;
        double[][] normTestLabels;

        private bool trained = false;
        private bool training = false;
        private bool doTraining = false;
        private Thread trainThread;

        private bool weightsSaved = false;

        int currentIteration = 0;
        int maxIterations = 10;
        int taskIterSize = 100;
        int numTrainImagesToEncode = 1000;
        int numTestImagesToEncode = 10000;
        double learningRate = 0.1;

        private string filePath;
        private string fileName;

        private Hashtable mnistTextures = new Hashtable();

        private long trainingStartTime = 0;

        int testIndex = -1;

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
            ReadMNISTTestData();
            normTrainingImages = EncodeImages(numTrainImagesToEncode, false);
            normTrainingLabels = OneHotEncodeLabels(numTrainImagesToEncode, false);
            normTestImages = EncodeImages(numTestImagesToEncode, true);
            normTestLabels = OneHotEncodeLabels(numTestImagesToEncode, true);
        }

        public void DisplayMNISTImage(int idx, TextureRect rect, bool testOrTrain)
        {
            byte[,,] images = null;
            if(testOrTrain)
            {
                images = testImages;
            }
            else
            {
                images = trainingImages;
            }
            Texture2D tex = null;
            if(mnistTextures.ContainsKey(testOrTrain.ToString() + "-" + idx))
            {
                tex = (Texture2D)mnistTextures[testOrTrain.ToString() + "-" + idx];
            }
            else
            {
                Image img = ConvertMNISTToImage(images, idx);
                tex = ImageTexture.CreateFromImage(img);
                mnistTextures[testOrTrain.ToString() + "-" + idx] = tex;
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


            //no thread training?
            if (trainThread == null && currentIteration < maxIterations)
            {
                trainThread = new Thread(new ParameterizedThreadStart(neuralNet.Train));
                Object trainingParams = new Object[] { normTrainingImages, normTrainingLabels, taskIterSize, learningRate };
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
                            saveWeightsButton.Disabled = false;
                            readWeightsButton.Disabled = false;
                            testButton.Disabled = false;
                            trainButton.Text = "Start Training";
                            doTraining = false;
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

                        progressLabel.Text = $"NumImagesToTrain: {numTrainImagesToEncode}\nTotalIterations: {maxIterations * taskIterSize}\n\nTraining: Iteration {currentIteration}/{maxIterations}\nThreadIteration: {currentThreadIteration + 1}/{taskIterSize}\nSample: {currentThreadInputIndex + 1}/{normTrainingImages.Length}\nAvgError: {currentThreadError:F10}\nMaxError: {currentThreadMaxError:F10}\nPercentComplete: {percentageComplete:F1}\nEstTotalTime (s): {estimatedTotalTime:F3}\nTimeLeft (s): {timeLeft:F3}";

                        DisplayMNISTImage(currentThreadInputIndex, trainRect, false);
                        DisplayMNISTImage(currentThreadMaxErrorIndex, maxErrorRect, false);
                    }
                }
            }
        }

        private double[][] OneHotEncodeLabels(int numEncode, bool testOrTrain)
        {
            double[][] normLabels = new double[numEncode][];
            for (int i = 0; i < numEncode; i++)
            {
                double[] oneHotVector = new double[10];
                if (testOrTrain)
                    oneHotVector[testLabels[i]] = 1.0;
                else
                    oneHotVector[trainingLabels[i]] = 1.0;
                normLabels[i] = oneHotVector;
            }
            return normLabels;
        }

        private double[][] EncodeImages(int numEncode, bool testOrTrain)
        {
            double[][] normImages = new double[numEncode][];
            for (int i = 0; i < numEncode; i++)
            {
                double[] imageVector = new double[784]; // 28x28 = 784 pixels
                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        if(testOrTrain)
                            imageVector[y * 28 + x] = testImages[i, y, x] / 255.0;
                        else
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

        private void ReadMNISTTestData()
        {
            testImages = (byte[,,])IDXReader.ReadIDX("MNIST/t10k-images.idx3-ubyte");
            testLabels = (byte[])IDXReader.ReadIDX("MNIST/t10k-labels.idx1-ubyte");
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
                testButton.Disabled = false;
            }
            else
            {
                doTraining = true;
                trainButton.Text = "Stop Training";
                readWeightsButton.Disabled = true;
                saveWeightsButton.Disabled = true;
                testButton.Disabled = true;
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

        private void OnTestLineEditSubmitted(string testLine)
        {
            testIndex = int.Parse(testLine);
            DisplayMNISTImage(testIndex, testRect, true);
        }

        private double TestImage(bool display)
        {
            double[] outputs = neuralNet.GetOutputs(normTestImages[testIndex]);
            int predictedNum = -1;
            double maxProb = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] > maxProb)
                {
                    maxProb = outputs[i];
                    predictedNum = i;
                }
            }
            if(display)
                testLabel.Text = $"Predicted: {predictedNum} (Prob: {maxProb:F4})\nActual: {testLabels[testIndex]}";
            return predictedNum;
        }

        private void OnTestButtonPressed()
        {
            if(testIndex != -1)
            {
                TestImage(true);
            }
        }

        private void OnTestAllButtonPressed()
        {
            int numCorrect = 0;
            for (int i = 0; i < normTestImages.Length; i++)
            {
                testIndex = i;
                double predictedNum = TestImage(false);
                if (predictedNum == testLabels[i])
                {
                    numCorrect++;
                }
            }
            numCorrectLabel.Text = $"Number of correct predictions: {numCorrect}/{normTestImages.Length}";
        }
    }
}
