using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Backpropagating_Neural_Network
{
    public partial class Form1 : Form
    {
        public static int generation = 1;
        private static Random R;
        public const int tests = 100;
        private NeuralNetwork net;
        private Timer updatePopulationTimer;
        private string trainingDataPath;
        private float learningRate = 0.25f;
        private float lambda = 5f;
        List<float> costs = new List<float>();
        List<float> percents = new List<float>();
        Bitmap[] batch = new Bitmap[tests];
        int[] nums = new int[tests];
        string[] files;
        public Form1()
        {
            InitializeComponent();
            typeof(Form1).InvokeMember("DoubleBuffered",
                BindingFlags.SetProperty | BindingFlags.Instance | BindingFlags.NonPublic, null, this,
                new object[] { true });

            R = new Random();
            net = new NeuralNetwork(tests, 20, 16, 10);
            updatePopulationTimer = new Timer
            {
                Enabled = true,
                Interval = 1
            };
            updatePopulationTimer.Tick += UpdateNet;
            trainingDataPath = Path.Combine(Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, @"..\..\..")), @"Images\");
            costs.Add(0.75f);
            percents.Add(0.1f);
            files = Directory.GetFiles(trainingDataPath, "*.png", SearchOption.AllDirectories);
        }
        private void UpdateNet(object sender, EventArgs e)
        {
            float multiplier = 0.01f;
            for (int j = 0; j < 10; j++)
            {
                for (int i = 0; i < batch.Length; i++)
                {
                    int number = R.Next(files.Length);
                    string randomTrainingDataPath = files[number];
                    Bitmap randomTrainingData = new Bitmap(randomTrainingDataPath);
                    batch[i] = randomTrainingData;
                    nums[i] = int.Parse(Directory.GetParent(randomTrainingDataPath).Name);
                    net.BackpropagateNet(batch[i], nums[i], learningRate, i);
                }
                net.MakeChanges(lambda);
                costs.Add((costs[costs.Count - 1] + net.totalCost / batch.Length / 5 * multiplier) / (1 + multiplier));
                percents.Add((percents[percents.Count - 1] + net.totalRight / batch.Length * multiplier)/ (1 + multiplier));
                net.testsDone = 0;
                net.totalCost = 0;
                net.totalRight = 0;
                generation++;
            }
            generation--;
            Invalidate();
        }
        public float[] Normalize(float[] data)
        {
            float average = 0;
            float standDev = 0;
            for (int i = 0; i < data.Length; i++)
            {
                average += data[i];
            }
            average /= data.Length;
            for (int i = 0; i < data.Length; i++)
            {
                standDev += (data[i] - average) * (data[i] - average);
            }
            standDev /= data.Length;
            standDev = (float)Math.Sqrt(standDev);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] -= average;
                data[i] /= standDev;
            }
            return data;
        }
        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.HighQuality;
            int graphWidth = 750;
            int paddingSize = 10;
            net.Draw(e.Graphics, new Rectangle(0,0, ClientRectangle.Width - graphWidth - 2 * paddingSize, ClientRectangle.Height));
            if (costs.Count > 1)
            {
                DrawGraph(e.Graphics, new Rectangle(ClientRectangle.Width - graphWidth - paddingSize, paddingSize, graphWidth, ClientRectangle.Height - 2 * paddingSize), costs, percents);
            }
        }
        public static float Bound(float f, float min, float max)
        {
            return Math.Max(Math.Min(f, max), min);
        }
        public static void DrawGraph(Graphics g, RectangleF Bounds, List<float> costs, List<float> percents)
        {
            PointF origin = new PointF(Bounds.Left + 1, Bounds.Bottom - 1);
            SizeF graphSize = new SizeF(Bounds.Width - 2, Bounds.Height - 2);
            g.DrawRectangle(Pens.White, Bounds.Left, Bounds.Top, Bounds.Width, Bounds.Height);
            //g.DrawLine(Pens.Red, origin, new Point(origin.X + graphSize.Width / scores.Count, (int)(origin.Y + graphSize.Height * scores[0] / 100)));
            float lineNum = Bounds.Width;
            float offset = Bound(costs.Count - 1 - lineNum, 1, costs.Count);
            float percentMultiplier = 10;
            float costMultiplier = 30;
            for (int i = 0; i < lineNum; i++)
            {
                if (i + offset < costs.Count)
                {
                    int diff = (int)((Bound((costs[(int)(i + offset)] - costs[(int)(i + offset - 1)]) * costMultiplier, -1, 1) + 1) / 2 * 255);
                    g.DrawLine(new Pen(Color.FromArgb(diff, 64, 255 - diff)),
                        new PointF(
                            origin.X + graphSize.Width * i / (costs.Count - offset),
                            (int)(origin.Y - graphSize.Height * costs[(int)(i + offset - 1)])),
                        new PointF(
                            origin.X + graphSize.Width * (i + 1) / (costs.Count - offset),
                            (int)(origin.Y - graphSize.Height * costs[(int)(i + offset)])));
                }
            }
            for (int i = 0; i < lineNum; i++)
            {
                if (i + offset < costs.Count)
                {
                    int diff = (int)((Bound((percents[(int)(i + offset)] - percents[(int)(i + offset - 1)]) * percentMultiplier, -1, 1) + 1) / 2 * 255);
                    g.DrawLine(new Pen(Color.FromArgb(255 - diff, diff, 0)),
                        new PointF(
                            origin.X + graphSize.Width * i / (costs.Count - offset),
                            (int)(origin.Y - graphSize.Height * percents[(int)(i + offset - 1)])),
                        new PointF(
                            origin.X + graphSize.Width * (i + 1) / (costs.Count - offset),
                            (int)(origin.Y - graphSize.Height * percents[(int)(i + offset)])));
                }
            }
            for (int i = 0; i < 100; i++)
            {
                g.DrawLine(Pens.White, Bounds.Right, Bounds.Bottom - Bounds.Height * i / 100.0f, Bounds.Right - (i % 10 == 0 ? 20 : (i % 5 == 0 ? 15 : 10)), Bounds.Bottom - Bounds.Height * i / 100.0f);
            }
        }
        private void Form1_KeyDown(object sender, KeyEventArgs e)
        {
            if(e.KeyCode == Keys.Space)
            {
                for(int i = 0; i < net.neurons.Length - 1; i++)
                {
                    for (int j = 0; j < net.neurons[i].Length; j++)
                    {
                        net.neurons[i][j].ReRandomizeIfApplicable(R, net.neurons[i+1], j);
                    }
                }
            }
        }
    }
    public struct NeuralNetwork
    {
        public static readonly Color positiveColor = Color.FromArgb(239, 100, 79);
        public static readonly Color negativeColor = Color.FromArgb(107, 179, 197);
        public const int imageWidth = 28;
        public const int imageHeight = 28;
        public static Random mutationRandom;
        public Neuron[][] neurons;
        public int testsDone;
        public float totalCost;
        public float totalRight;
        private Bitmap current;
        public byte[][] firstLayerFilterBytes;
        public Bitmap[] firstLayerFilters;
        public ColorPalette colorPallette;
        private readonly int[] correct;
        private float[][] inputs;
        private readonly float[] outputs;
        private const int mutationFrequency = 15;
        public NeuralNetwork(int tests, params int[] layersizes)
        {
            mutationRandom = new Random();
            neurons = new Neuron[layersizes.Length][];
            inputs = new float[neurons.Length + 1][];
            testsDone = 0;
            totalCost = 0;
            totalRight = 0;
            current = new Bitmap(imageWidth, imageHeight);
            correct = new int[10];
            outputs = new float[layersizes[layersizes.Length - 1]];
            int layerSize = imageWidth * imageHeight;
            for (int i = 0; i < layersizes.Length; i++)
            {
                neurons[i] = new Neuron[layersizes[i]];
                int prevLayerSize = layerSize;
                layerSize = layersizes[i];
                for (int j = 0; j < layerSize; j++)
                {
                    string neuronString = string.Empty;
                    for (int k = 0; k < prevLayerSize + 1; k++)
                    {
                        neuronString += ((mutationRandom.NextDouble() - 0.5)*0.5).ToString("0.00000") + ' ';
                    }
                    neurons[i][j] = new Neuron(neuronString, tests);
                }
            }
            firstLayerFilterBytes = new byte[layersizes[0]][];
            for (int i = 0; i < layersizes[0]; i++)
            {
                firstLayerFilterBytes[i] = new byte[imageWidth * imageHeight];
                for(int j = 0; j < imageWidth * imageHeight; j++)
                {
                    firstLayerFilterBytes[i][j] = (byte)(Bound((neurons[0][i].weights[j] + 1) / 2, 0, 1) * byte.MaxValue);
                }
            }
            firstLayerFilters = new Bitmap[layersizes[0]];
            for(int i = 0; i < layersizes[0]; i++)
            {
                firstLayerFilters[i] = new Bitmap(imageWidth, imageHeight, imageWidth, PixelFormat.Format8bppIndexed, Marshal.UnsafeAddrOfPinnedArrayElement(firstLayerFilterBytes[i], 0));
            }
            colorPallette = firstLayerFilters[0].Palette;
            for (int i = 0; i < colorPallette.Entries.Length / 2; i++)
            {
                float multiplier = 1 - (i / (float)(colorPallette.Entries.Length / 2));
                colorPallette.Entries[i] = Color.FromArgb((int)(negativeColor.R * multiplier), (int)(negativeColor.G * multiplier), (int)(negativeColor.B * multiplier));
            }
            for (int i = 0; i < colorPallette.Entries.Length / 2; i++)
            {
                float multiplier = (i / (float)(colorPallette.Entries.Length / 2));
                colorPallette.Entries[i + colorPallette.Entries.Length / 2] = Color.FromArgb((int)(positiveColor.R * multiplier), (int)(positiveColor.G * multiplier), (int)(positiveColor.B * multiplier));
            }
            for (int i = 0; i < layersizes[1]; i++)
            {
                firstLayerFilters[i].Palette = colorPallette;
            }
        }
        public static float Bound(float f, float min, float max)
        {
            return Math.Max(Math.Min(f, max), min);
        }
        public float[] Normalize(float[] data)
        {
            float average = 0;
            float standDev = 0;
            for(int i = 0; i < data.Length; i++)
            {
                average += data[i];
            }
            average /= data.Length;
            for (int i = 0; i < data.Length; i++)
            {
                standDev += (data[i] - average) * (data[i] - average);
            }
            standDev /= data.Length;
            standDev = (float)Math.Sqrt(standDev);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] -= average;
                data[i] /= standDev;
            }
            return data;
        }
        public float[] GetOutputs(Bitmap input)
        {
            inputs[0] = new float[imageWidth * imageHeight];
            for (int i = 0; i < imageHeight; i++)
            {
                for (int j = 0; j < imageHeight; j++)
                {
                    inputs[0][i * imageWidth + j] = input.GetPixel(j, i).GetBrightness() / 256.0f;
                }
            }
            inputs[0] = Normalize(inputs[0]);
            for (int i = 0; i < neurons.Length; i++)
            {
                if (i > 0)
                    inputs[i] = Normalize(inputs[i]);
                inputs[i][inputs[i].Length - 1] = 1;
                if (i < neurons.Length - 1)
                    inputs[i + 1] = new float[neurons[i].Length + 1];
                else
                    inputs[i + 1] = new float[neurons[i].Length];
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    inputs[i + 1][j] = neurons[i][j].GetOutput(inputs[i]);
                }
            }
            inputs[inputs.Length - 1] = Normalize(inputs[inputs.Length - 1]);
            return inputs[inputs.Length - 1];
        }
        public int MostConfidentOutput(float[] outputs)
        {
            return outputs.ToList().IndexOf(outputs.Max());
        }
        public float GetCost(int desired, float[] output)
        {
            float totalCost = 0;
            float[] newOutput = new float[output.Length];
            for (int i = 0; i < output.Length; i++)
            {
                newOutput[i] = Bound(output[i], -0.105409255339f, 0.948683298051f);
                if (i == desired)
                {
                    totalCost += (0.948683298051f - newOutput[i]) * (0.948683298051f - newOutput[i]);
                }
                else
                {
                    totalCost += (0.105409255339f + newOutput[i]) * (0.105409255339f + newOutput[i]);
                }
            }
            return totalCost;
        }
        public void Draw(Graphics g, Rectangle bounds)
        {
            if (inputs[1] == null)
                return;
            float imageDrawWidth = bounds.Height / (2.0f * neurons[0].Length + 1) * 4 - 2 * Neuron.thresholdEllipseOutlineWidth;
            float paddingLeft = 20 + imageDrawWidth * 3.5f;
            float paddingRight = 50;
            SizeF spacing = new SizeF((bounds.Width - paddingLeft - paddingRight) / (neurons.Length - 1), bounds.Height / (2.0f * neurons[0].Length + 1));
            float spacingA = 0;
            for (int i = 0; i < neurons.Length; i++)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    if (i > 0)
                    {
                        spacingA = (bounds.Height - spacing.Height * (2 * neurons[i].Length - 1)) / 2;
                        float spacingB = (bounds.Height - spacing.Height * (2 * neurons[i - 1].Length - 1)) / 2;
                        for (int k = 0; k < neurons[i - 1].Length + 1; k++)
                        {
                            float brightness = Bound(Math.Abs(neurons[i][j].weights[k]), 0, 1);
                            g.DrawLine(
                                new Pen(
                                    neurons[i][j].weights[k] >= 0 ?
                                        positiveColor :
                                        negativeColor,brightness*3
                                    ),
                                paddingLeft + i * spacing.Width,
                                spacingA + (bounds.Height - 2.0f * spacing.Height) / neurons[0].Length * j,
                                paddingLeft + (i - 1) * spacing.Width,
                                spacingB + (bounds.Height - 2.0f * spacing.Height) / neurons[0].Length * k);
                        }
                    }
                }
            }
            for (int i = 0; i < inputs.Length - 1; i++)
            {
                spacingA = (bounds.Height - spacing.Height * (2 * neurons[i].Length - 1)) / 2;
                for (int j = 0; j < inputs[i + 1].Length; j++)
                {
                    g.FillEllipse(
                        Brushes.White,
                        paddingLeft + i * spacing.Width - Neuron.thresholdEllipseDiameter / 2 - Neuron.thresholdEllipseOutlineWidth,
                        spacingA + (bounds.Height - 2.0f * spacing.Height) / neurons[0].Length * j - Neuron.thresholdEllipseDiameter / 2 - Neuron.thresholdEllipseOutlineWidth,
                        Neuron.thresholdEllipseDiameter + Neuron.thresholdEllipseOutlineWidth * 2,
                        Neuron.thresholdEllipseDiameter + Neuron.thresholdEllipseOutlineWidth * 2);
                    g.FillEllipse(
                        new SolidBrush(
                            Color.FromArgb(
                                (int)Bound((inputs[i + 1][j] * 128), 0, 255),
                                (int)Bound((inputs[i + 1][j] * 128), 0, 255),
                                (int)Bound((inputs[i + 1][j] * 128), 0, 255)
                            )
                        ),
                        paddingLeft + i * spacing.Width - Neuron.thresholdEllipseDiameter / 2,
                        spacingA + (bounds.Height - 2.0f * spacing.Height) / neurons[0].Length * j - Neuron.thresholdEllipseDiameter / 2,
                        Neuron.thresholdEllipseDiameter,
                        Neuron.thresholdEllipseDiameter);
                }
            }
            for (int j = 0; j < neurons[0].Length; j++)
            {
                g.FillRectangle(
                    Brushes.White,
                    paddingLeft - (imageDrawWidth * (j % 2 + 1)) - Neuron.thresholdEllipseDiameter / 2 - Neuron.thresholdEllipseOutlineWidth * 3 - Neuron.thresholdEllipseOutlineWidth * (j % 2) - 5,
                    spacing.Height + (bounds.Height - 2.0f * spacing.Height) / neurons[0].Length * j - imageDrawWidth / 2 - Neuron.thresholdEllipseOutlineWidth,
                    imageDrawWidth + Neuron.thresholdEllipseOutlineWidth * 2,
                    imageDrawWidth + Neuron.thresholdEllipseOutlineWidth * 2);
                g.DrawImage(
                    firstLayerFilters[j],
                    paddingLeft - (imageDrawWidth * (j % 2 + 1)) - Neuron.thresholdEllipseDiameter / 2 - Neuron.thresholdEllipseOutlineWidth * 2 - Neuron.thresholdEllipseOutlineWidth * (j % 2) - 5,
                    spacing.Height + (bounds.Height - 2.0f * spacing.Height) / neurons[0].Length * j - imageDrawWidth / 2,
                    imageDrawWidth,
                    imageDrawWidth);
            }
            spacingA = (bounds.Height - spacing.Height * (2 * neurons[neurons.Length - 1].Length - 1)) / 2;
            for (int j = 0; j < neurons[neurons.Length - 1].Length; j++)
            {
                g.FillEllipse(
                    Brushes.White,
                    paddingLeft + (neurons.Length - 1) * spacing.Width + Neuron.thresholdEllipseDiameter + Neuron.thresholdEllipseOutlineWidth * 2 + 5 - Neuron.thresholdEllipseDiameter / 2 - Neuron.thresholdEllipseOutlineWidth,
                    spacingA + (bounds.Height - 2.0f * spacing.Height) / neurons[0].Length * j - Neuron.thresholdEllipseDiameter / 2 - Neuron.thresholdEllipseOutlineWidth,
                    Neuron.thresholdEllipseDiameter + Neuron.thresholdEllipseOutlineWidth * 2,
                    Neuron.thresholdEllipseDiameter + Neuron.thresholdEllipseOutlineWidth * 2);
                g.FillEllipse(
                    new SolidBrush(
                        Color.FromArgb(
                            (int)Bound((correct[j]) * 256, 0, 255),
                            (int)Bound((correct[j]) * 256, 0, 255),
                            (int)Bound((correct[j]) * 256, 0, 255)
                        )
                    ),
                    paddingLeft + (neurons.Length - 1) * spacing.Width + Neuron.thresholdEllipseDiameter + Neuron.thresholdEllipseOutlineWidth * 2 + 5 - Neuron.thresholdEllipseDiameter / 2,
                    spacingA + (bounds.Height - 2.0f * spacing.Height) / neurons[0].Length * j - Neuron.thresholdEllipseDiameter / 2,
                    Neuron.thresholdEllipseDiameter,
                    Neuron.thresholdEllipseDiameter);
            }
            //g.DrawString("" + (int)(totalCost / testsDone * 100) / 10.0 + "% correct", new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 10);
            g.DrawString("Generation " + Form1.generation, new Font(FontFamily.GenericMonospace, 20), Brushes.White, (bounds.Width - g.MeasureString("Generation " + Form1.generation, new Font(FontFamily.GenericMonospace, 20)).Width) / 2, 10);
            g.DrawImage(current,10, bounds.Height / 2 - imageDrawWidth / 2, imageDrawWidth, imageDrawWidth);
            Form1.generation++;
        }
        public void BackpropagateNet(Bitmap current, int correct, float learningRate, int test)
        {
            this.current = current;
            float[] outputsGotten = GetOutputs(current);
            for (int i = 0; i < neurons[neurons.Length - 1].Length; i++)
            {
                this.correct[i] = i == correct ? 1 : 0;
                neurons[neurons.Length - 1][i].CalculateErrorSignal(i == correct ? 1 : -0);
                neurons[neurons.Length - 1][i].StoreChanges(learningRate, test);// * Neuron.LnMirrored(Math.Min(Form1.generation, 60) / 27.183f));
            }
            for (int i = neurons.Length - 2; i >= 0; i--)
            {
                for (int j = 0; j < neurons[i].Length; j++)
                {
                    neurons[i][j].CalculateErrorSignal(neurons[i + 1], j);
                    neurons[i][j].StoreChanges(learningRate, test);// * Neuron.LnMirrored(Math.Min(Form1.generation, 60) / 27.183f));
                }
            }
            totalCost += GetCost(correct, outputsGotten);
            totalRight += MostConfidentOutput(outputsGotten) == correct ? 1 : 0;
        }
        public void MakeChanges(float lambda)
        {
            foreach (Neuron[] nl in neurons)
            {
                foreach (Neuron n in nl)
                {
                    n.MakeChanges(lambda);
                }
            }
            for (int i = 0; i < inputs[1].Length - 1; i++)
            {
                for (int j = 0; j < inputs[0].Length - 1; j++)
                {
                    firstLayerFilterBytes[i][j] = (byte)(Bound((neurons[0][i].weights[j] + 1) / 2, -1, 1) * byte.MaxValue);
                }
            }
            for (int i = 0; i < inputs[1].Length - 1; i++)
            {
                firstLayerFilters[i] = new Bitmap(imageWidth, imageHeight, imageWidth, PixelFormat.Format8bppIndexed, Marshal.UnsafeAddrOfPinnedArrayElement(firstLayerFilterBytes[i], 0))
                {
                    Palette = colorPallette
                };
            }
        }
    }
    public struct Neuron
    {
        public const float maxWeightLineWidth = 2;
        public const float thresholdEllipseDiameter = 20;
        public const float thresholdEllipseOutlineWidth = 1;
        public string DNA;
        public float errorSignal;
        public float[][] currentBatchWeightsNudges;
        public float[] weights;
        public float[] input;
        public float weightedSum;
        public float output;
        public bool finalLayer;
        public Neuron(string gene, int tests, bool isFinalLayer = false)
        {
            DNA = gene;
            weights = new float[gene.Where(c => c == ' ').Count()];
            int weight = 0;
            errorSignal = 0;
            currentBatchWeightsNudges = new float[tests][];
            while (gene.Contains(' '))
            {
                weights[weight] = float.Parse(gene.Substring(0, gene.IndexOf(' ')));
                weight++;
                gene = gene.Substring(gene.IndexOf(' ') + 1);
            }
            input = new float[weight];
            weightedSum = 0;
            output = 0;
            finalLayer = isFinalLayer;
        }
        public float GetOutput(float[] inputs)
        {
            Array.Copy(inputs, input, inputs.Length);
            weightedSum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                weightedSum += input[i] * weights[i];
            }
            output = ActivationFunction(weightedSum);
            return output;
        }
        public void ReRandomizeIfApplicable(Random r, Neuron[] nextLayer, int index)
        {
            float average = 0;
            float standDev = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                average += weights[i];
            }
            average /= weights.Length;
            for (int i = 0; i < weights.Length; i++)
            {
                standDev += (weights[i] - average) * (weights[i] - average);
            }
            standDev /= weights.Length;
            if (Math.Abs(standDev) >= 0.0005)
                return;
            for(int i = 0; i < weights.Length; i++)
            {
                weights[i] = (float)r.NextDouble() - 0.5f;
            }
            for(int i = 0; i < nextLayer.Length; i++)
            {
                nextLayer[i].weights[index] = (float)r.NextDouble() - 0.5f;
            }
        }
        float ActivationFunction(float input)
        {
            return finalLayer ? Tanh(input) : SoftPlus(input);
        }
        float ActivationFunctionDerivative(float input)
        {
            return finalLayer ? TanhDerivative(input) : SoftPlusDerivative(input);
        }
        public static float LnMirrored(float input)
        {
            return (float)Math.Log(Math.Abs(input) + 1);
        }
        public static float LnMirroredDerivative(float input)
        {
            return (float)(1/ (Math.Abs(input) + 1));
        }
        public static float SigmoidDerivative(float input)
        {
            double ex = Math.Pow(2.718281828459, input);
            return (float)(ex / (1 + ex) / (1 + ex));
        }
        public static float TanhDerivative(float input)
        {
            double tanh = Math.Tanh(input);
            return (float)(1 - tanh * tanh);
        }
        public static float ReLUDerivative(float input)
        {
            return input > 0 ? 1 : 0;
        }
        public static float SoftPlusDerivative(float input)
        {
            return (float)(1 / (1 + Math.Pow(2.718281828459, -input)));
        }
        public static float Sigmoid(float input)
        {
            return (float)(1 / (1 + Math.Pow(2.718281828459, -input)));
        }
        public static float Tanh(float input)
        {
            return (float)Math.Tanh(input);
        }
        public static float ReLU(float input)
        {
            return input > 0 ? input : 0;
        }
        public static float SoftPlus(float input)
        {
            return (float)Math.Log(1 + Math.Pow(2.718281828459, input));
        }
        public void CalculateErrorSignal(float desiredOutput)
        {
            errorSignal = (output - desiredOutput) * ActivationFunctionDerivative(weightedSum);
        }
        public void CalculateErrorSignal(Neuron[] nextLayer, int index)
        {
            errorSignal = 0;
            for (int i = 0; i < nextLayer.Length; i++)
            {
                errorSignal += nextLayer[i].errorSignal * nextLayer[i].weights[index];
            }
            errorSignal *= ActivationFunctionDerivative(weightedSum);
        }
        public void StoreChanges(float learningRate, int test)
        {
            currentBatchWeightsNudges[test] = new float[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                currentBatchWeightsNudges[test][i] = Bound((input[i] * errorSignal) * learningRate, -1, 1);
            }
        }
        public static float Bound(float f, float min, float max)
        {
            return Math.Max(Math.Min(f, max), min);
        }
        public void MakeChanges(float lambda)
        {
            float[] weightChanges = new float[currentBatchWeightsNudges[0].Length];
            for (int j = 0; j < weightChanges.Length; j++)
            {
                float total = 0;
                for (int k = 0; k < currentBatchWeightsNudges.Length; k++)
                    total += currentBatchWeightsNudges[k][j];
                total += weights[j] * lambda / currentBatchWeightsNudges.Length;
                weightChanges[j] = total / currentBatchWeightsNudges.Length;
            }
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = Bound(weights[i] - weightChanges[i], -1, 1);
            }
        }
    }
}