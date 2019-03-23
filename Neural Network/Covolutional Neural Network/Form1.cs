using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Covolutional_Neural_Network
{
    public partial class Form1 : Form
    {
        public static int generation = 0;
        public static Random R;
        //const int bestPopulation = 6;
        private const int populationSize = 100;
        private const int tests = 1000;
        private ConvNet[] population;
        private Timer updatePopulationTimer;
        //private string trainingDataPath;
        ConvNet best;
        List<float> scores = new List<float>();
        //private string[] files;
        private float[][][][] allNeuronWeights;
        private Neuron[][][] allNeurons;
        private float[][][] allInputs;
        private int[] layersizes;
        private Digit[] allDigits;
        public Form1()
        {
            InitializeComponent();
            typeof(Form1).InvokeMember("floatBuffered",
                BindingFlags.SetProperty | BindingFlags.Instance | BindingFlags.NonPublic, null, this,
                new object[] { true });
            string[] files = Directory.GetFiles(Path.Combine(Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, @"..\..\..")), @"Images\"), "*.bmp", SearchOption.AllDirectories);
            allDigits = new Digit[files.Length];
            for (int i = 0; i < files.Length; i++)
            {
                allDigits[i] = new Digit(files[i]);
            }
            layersizes = new[] { 137, 57, 24, 10 };
            allNeuronWeights = new float[populationSize][][][];
            allNeurons = new Neuron[populationSize][][];
            allInputs = new float[populationSize][][];
            for (int i = 0; i < populationSize; i++)
            {
                allNeuronWeights[i] = new float[layersizes.Length][][];
                for (int j = 0; j < layersizes.Length; j++)
                {
                    allNeuronWeights[i][j] = new float[layersizes[j]][];
                    allNeuronWeights[i][j][0] = new float[784];
                    for (int k = 1; k < layersizes[j]; k++)
                    {
                        allNeuronWeights[i][j][k] = new float[layersizes[k - 1]];
                    }
                }
            }
            R = new Random();
            population = new ConvNet[populationSize];
            for (int i = 0; i < populationSize; i++)
            {
                population[i] = new ConvNet(i, allNeuronWeights, allNeurons, allInputs);
            }
            updatePopulationTimer = new Timer()
            {
                Enabled = true,
                Interval = 1
            };
            updatePopulationTimer.Tick += UpdatePopulation;
            scores.Add(10);
        }
        [GpuManaged]
        private void UpdatePopulation(object sender, EventArgs e)
        {
            float[][] inputs = new float[tests][];
            string[] paths = new string[tests];
            float[][] outputs = new float[populationSize * tests][];
            ConvNet[] functionPopulation = population.Select(n => n).ToArray();
            float[][][][] functionNeuronWeights = allNeuronWeights.Select(n => n).ToArray();
            Neuron[][][] functionNeurons = allNeurons.Select(n => n).ToArray();
            float[][][] functionInputs = allInputs.Select(n => n).ToArray();
            int[] values = new int[tests];
            for (int i = 0; i < tests; i++)
            {
                int place = R.Next(allDigits.Length);
                values[i] = allDigits[place].value;
                inputs[i] = allDigits[place].pixelValues;
                for (int j = 0; j < populationSize; j++)
                {
                    outputs[i * populationSize + j] = new float[10];
                }
            }
            Gpu.Default.For(0, population.Length * tests, i =>
            {
                outputs[i] = functionPopulation[i / tests].GetOutputs(inputs[i % tests], functionNeuronWeights, functionNeurons, functionInputs);
            }
            );
            for (int i = 0; i < populationSize * tests; i++)
            {
                ConvNet nn = functionPopulation[i / tests];
                nn.sumCost = nn.sumCost + nn.GetCost(values[i % tests], outputs[i]);
                nn.testsDone++;
                if (nn.MostConfidentOutput(outputs[i]) == values[i % tests])
                {
                    nn.correct++;
                }
                functionPopulation[i / tests] = nn;
            }
            population = functionPopulation.Select(n => n).ToArray();
            allNeuronWeights = functionNeuronWeights.Select(n => n).ToArray();
            allNeurons = functionNeurons.Select(n => n).ToArray();
            allInputs = functionInputs.Select(n => n).ToArray();
            float[][][][] newNeuronWeights = new float[allNeuronWeights.Length][][][];
            Neuron[][][] newNeurons = new Neuron[allNeurons.Length][][];
            float[][][] newInputs = new float[allInputs.Length][][];
            List<float> grades = new List<float>();
            foreach (ConvNet n in population)
            {
                grades.Add(n.sumCost);
            }
            Array.Sort(grades.ToArray(), population);
            //population = population.Reverse().ToArray();
            List<ConvNet> topScorers = new List<ConvNet>();
            List<int> order = Randomize(Enumerable.Range(0, populationSize / 2).ToList(), R);
            int numRemaining = populationSize / 2;
            for (int i = populationSize - 1; i >= 0; i--)
            {
                if (R.Next(populationSize) >= i || numRemaining == 0)
                    topScorers.Add(population[i]);
                else
                    numRemaining--;
            }
            topScorers.RemoveRange(0, numRemaining);
            List<ConvNet> newPopulation = new List<ConvNet>();
            for (int i = 0; newPopulation.Count < populationSize; i++)
            {
                for (int j = 0; j < 7 && newPopulation.Count < populationSize; j++)
                {
                    if (j >= 2)
                        if (R.Next(order[i]) == 0)
                            newPopulation.Add(new ConvNet(topScorers[order[i]], topScorers[order[(i + j) % order.Count]], newPopulation.Count, newNeuronWeights, newNeurons, newInputs, allNeuronWeights, allNeurons, allInputs));
                        else
                            newPopulation.Add(new ConvNet(topScorers[order[i]], topScorers[order[(i + j) % order.Count]], newPopulation.Count, newNeuronWeights, newNeurons, newInputs, allNeuronWeights, allNeurons, allInputs));
                }
            }
            best = new ConvNet(population[0], 0, newNeuronWeights, newNeurons, newInputs, allNeuronWeights, allNeurons, allInputs);
            newPopulation[0] = best;
            population = newPopulation.ToArray();
            Array.Copy(newNeuronWeights, allNeuronWeights, 0);
            Array.Copy(newNeurons, allNeurons, 0);
            Array.Copy(newInputs, allInputs, 0);
            scores.Add(((float)best.correct / best.testsDone) * 100);
            Invalidate();
        }
        List<ConvNet> Randomize(List<ConvNet> sorted, Random r)
        {
            List<ConvNet> result = new List<ConvNet>();
            List<int> placesLeft = new List<int>();
            for (int i = 0; i < sorted.Count; i++)
            {
                placesLeft.Add(i);
            }
            while (placesLeft.Count > 0)
            {
                int placeTaken = r.Next(placesLeft.Count);
                result.Add(sorted[placesLeft[placeTaken]]);
                placesLeft.RemoveAt(placeTaken);
            }
            return result;
        }
        List<int> Randomize(List<int> sorted, Random r)
        {
            List<int> result = new List<int>();
            List<int> placesLeft = new List<int>();
            for (int i = 0; i < sorted.Count; i++)
            {
                placesLeft.Add(i);
            }
            while (placesLeft.Count > 0)
            {
                int placeTaken = r.Next(placesLeft.Count);
                result.Add(sorted[placesLeft[placeTaken]]);
                placesLeft.RemoveAt(placeTaken);
            }
            return result;
        }

        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            if (scores.Count > 1)
                DrawGraph(e.Graphics, new Rectangle(10, 90, 210, 140), scores);
            best.Draw(e.Graphics, ClientRectangle, allNeuronWeights, allNeurons);
        }
        public void DrawGraph(Graphics g, Rectangle Bounds, List<float> scores)
        {
            g.DrawRectangle(Pens.Black, Bounds);
            g.DrawLine(Pens.Black, new Point(Bounds.Left + 10, Bounds.Top + 10), new Point(Bounds.Left + 10, Bounds.Bottom - 10));
            g.DrawLine(Pens.Black, new Point(Bounds.Left + 10, Bounds.Bottom - 10), new Point(Bounds.Right - 10, Bounds.Bottom - 10));
            Point origin = new Point(Bounds.Left + 10, Bounds.Bottom - 10);
            Size graphSize = new Size(Bounds.Width - 20, Bounds.Height - 20);
            //g.DrawLine(Pens.Red, origin, new Point(origin.X + graphSize.Width / scores.Count, (int)(origin.Y + graphSize.Height * scores[0] / 100)));
            for (int i = 1; i < scores.Count; i++)
            {
                g.DrawLine(Pens.Red, new Point(origin.X + graphSize.Width * (i - 1) / (scores.Count - 1), (int)(origin.Y - graphSize.Height * scores[i - 1] / 100)), new Point(origin.X + graphSize.Width * i / (scores.Count - 1), (int)(origin.Y - graphSize.Height * scores[i] / 100)));
            }
        }
    }
    public struct ConvNet
    {
        public float avgCost;
        public const int imageWidth = 28;
        public const int imageHeight = 28;
        public static Random mutationRandom;
        public int testsDone;
        public int correct;
        private Bitmap current;
        public const int fullyConnectedLayers = 2;
        public const int pixelSize = 5;
        public ConvNeuron[][] featureMaps;
        public Neuron[][] fullyConnected;
        private Bitmap[][] inputs;
        private float[][] fullyConnectedInputs;
        private float[] outputs;
        private const int mutationFrequency = 15;
        public static bool operator >(ConvNet a, ConvNet b)
        {
            return a.correct > b.correct;
        }
        public static bool operator <(ConvNet a, ConvNet b)
        {
            return a.correct < b.correct;
        }
        public ConvNet(params int[] layerSizes)
        {
            mutationRandom = new Random();
            inputs = new Bitmap[layerSizes.Length - fullyConnectedLayers][];
            fullyConnectedInputs = new float[fullyConnectedLayers][];
            featureMaps = new ConvNeuron[layerSizes.Length - fullyConnectedLayers][];
            fullyConnected = new Neuron[fullyConnectedLayers][];
            testsDone = 0;
            correct = 0;
            avgCost = 0;
            current = new Bitmap(imageWidth, imageHeight);
            outputs = new float[layerSizes[layerSizes.Length - 1]];
            for (int i = 0; i < layerSizes[layerSizes.Length - 1]; i++)
            {
                outputs[i] = 0;
            }
            int prevLayerSize = 1;
            for (int i = 0; i < layerSizes.Length - fullyConnectedLayers; i++)
            {
                featureMaps[i] = new ConvNeuron[];
                int layerSize = layerSizes[i];
                prevLayerSize *= layerSize;
                for (int j = 0; j < layerSize; j++)
                {
                    string neuronString = (mutationRandom.Next(Neuron.divisor) + Neuron.divisor).ToString().Substring(1);
                    for (int k = 0; k < ConvNeuron.filterSize * ConvNeuron.filterSize; k++)
                    {
                        neuronString += (mutationRandom.Next(Neuron.divisor) + Neuron.divisor).ToString().Substring(1);
                    }
                    featureMaps[i].Add(new ConvNeuron(neuronString));
                }
            }
            for (int i = 0; i < fullyConnectedLayers; i++)
            {
                fullyConnected.Add(new List<Neuron>());
                for (int j = 0; j < layerSizes[layerSizes.Length - fullyConnectedLayers + i]; j++)
                {
                    string neuronString = (mutationRandom.Next(Neuron.divisor) + Neuron.divisor).ToString().Substring(1);
                    for (int k = 0; k < prevLayerSize; k++)
                    {
                        neuronString += (mutationRandom.Next(Neuron.divisor) + Neuron.divisor).ToString().Substring(1);
                    }
                    fullyConnected[i].Add(new Neuron(neuronString));
                }
                prevLayerSize = layerSizes[layerSizes.Length - fullyConnectedLayers + i];
            }
        }
        public ConvNet(ConvNet parentA, ConvNet parentB)
        {
            mutationRandom = new Random();
            inputs = new List<List<Bitmap>>();
            fullyConnectedInputs = new List<List<float>>();
            featureMaps = new List<List<ConvNeuron>>();
            fullyConnected = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            avgCost = 0;
            current = new Bitmap(28, 28);
            outputs = new List<float>();
            for (int i = 0; i < parentA.fullyConnected[1].Count; i++)
            {
                outputs.Add(0);
            }
            int prevLayerSize = 1;
            for (int i = 0; i < parentA.featureMaps.Count; i++)
            {
                featureMaps.Add(new List<ConvNeuron>());
                int layerSize = parentA.featureMaps[i].Count;
                prevLayerSize *= layerSize;
                for (int j = 0; j < layerSize; j++)
                {
                    string neuronADNA = parentA.featureMaps[i][j].DNA;
                    string neuronBDNA = parentB.featureMaps[i][j].DNA;
                    string crossoveredDNA = string.Empty;
                    for (int k = 0; k < neuronADNA.Length; k++)
                    {
                        crossoveredDNA += (mutationRandom.Next(2) == 0 ? neuronADNA : neuronBDNA)[k];
                        if (mutationRandom.Next(100 / mutationFrequency) == 0)
                            crossoveredDNA = crossoveredDNA.Substring(0, k) + mutationRandom.Next(10);
                    }
                    featureMaps[i].Add(new ConvNeuron(crossoveredDNA));
                }
            }
            for (int i = 0; i < fullyConnectedLayers; i++)
            {
                fullyConnected.Add(new List<Neuron>());
                int layerSize = parentA.fullyConnected[i].Count;
                for (int j = 0; j < layerSize; j++)
                {
                    string neuronADNA = parentA.fullyConnected[i][j].DNA;
                    string neuronBDNA = parentB.fullyConnected[i][j].DNA;
                    string crossoveredDNA = string.Empty;
                    for (int k = 0; k < neuronADNA.Length; k++)
                    {
                        crossoveredDNA += (mutationRandom.Next(2) == 0 ? neuronADNA : neuronBDNA)[k];
                        if (mutationRandom.Next(100 / mutationFrequency) == 0)
                            crossoveredDNA = crossoveredDNA.Substring(0, k) + mutationRandom.Next(10);
                    }
                    fullyConnected[i].Add(new Neuron(crossoveredDNA));
                }
                prevLayerSize = layerSize;
            }
        }
        public ConvNet(ConvNet original)
        {
            mutationRandom = new Random();
            inputs = original.inputs;
            fullyConnectedInputs = original.fullyConnectedInputs;
            featureMaps = original.featureMaps;
            fullyConnected = original.fullyConnected;
            testsDone = 0;
            correct = 0;
            avgCost = 0;
            current = new Bitmap(28, 28);
            outputs = original.outputs;
        }
        public List<float> getOutputs(Bitmap input)
        {
            inputs = new List<List<Bitmap>>();
            inputs.Add(new List<Bitmap>());
            inputs[0].Add(input);
            for (int i = 0; i < featureMaps.Count; i++)
            {
                inputs.Add(new List<Bitmap>());
                for (int j = 0; j < inputs[i].Count; j++)
                {
                    for (int k = 0; k < featureMaps[i].Count; k++)
                    {
                        inputs[i + 1].Add(ConvNeuron.Subsample(featureMaps[i][k].Convolve(inputs[i][j])));
                    }
                }
            }
            fullyConnectedInputs = new List<List<float>>();
            fullyConnectedInputs.Add(new List<float>());
            foreach (Bitmap b in inputs[inputs.Count - 1])
            {
                fullyConnectedInputs[0].Add(b.GetPixel(0, 0).GetBrightness() / 256.0);
            }
            for (int i = 0; i < fullyConnectedLayers; i++)
            {
                fullyConnectedInputs.Add(new List<float>());
                for (int j = 0; j < fullyConnected[i].Count; j++)
                {
                    fullyConnectedInputs[i + 1].Add(fullyConnected[i][j].GetOutput(fullyConnectedInputs[i].ToArray()));
                }
            }
            return fullyConnectedInputs[inputs.Count - 2];
        }
        public int mostConfidentOutput(List<float> outputs)
        {
            return outputs.ToList().IndexOf(outputs.Max());
        }
        public float getCost(int desired, List<float> output)
        {
            float totalCost = 0;
            for (int i = 0; i < output.Count; i++)
            {
                if (i == desired)
                {
                    totalCost += (1 - output[i]) * (1 - output[i]);
                }
                else
                {
                    totalCost += output[i] * output[i];
                }
            }
            return totalCost;
        }
        public Bitmap Scale(Bitmap input)
        {
            Bitmap output = new Bitmap(input.Width * 5, input.Height * 5);
            for(int i = 0; i < output.Width; i++)
            {
                for(int j = 0; j < output.Height; j++)
                {
                    output.SetPixel(i, j, input.GetPixel(i / 5, j / 5));
                }
            }
            return output;
        }
        public void Draw(Graphics g, Rectangle bounds)
        {
            if (featureMaps == null)
                return;
            int layers = 0;
            int imageWidth = 0;
            foreach (List<Bitmap> b in inputs)
            {
                imageWidth += b[0].Width * pixelSize;
                layers++;
            }
            imageWidth += featureMaps.Count * ConvNeuron.filterSize * pixelSize;
            layers += featureMaps.Count;
            imageWidth += fullyConnectedLayers * 5;
            layers += fullyConnectedLayers;
            Size padding = new Size(100, 100);
            int currentWidth = 0;
            for (int i = 0; i < featureMaps.Count; i++)
            {
                currentWidth += inputs[i][0].Width * pixelSize;
                for (int j = 0; j < featureMaps[i].Count; j++)
                {
                    for(int k = 0; k < inputs[i].Count; k++)
                    {
                        g.DrawLine(Pens.Black, 
                            padding.Width * (1 + 2 * i) - inputs[i][k].Width * pixelSize / 2 + currentWidth,
                            padding.Height + (int)((bounds.Height - 2 * padding.Height) / (float)(inputs[i].Count + 1) * (k + 1) - inputs[i][k].Height * pixelSize / 2) + inputs[i][k].Height * pixelSize / 2,
                            padding.Width * (2 + 2 * i) + currentWidth + ConvNeuron.filterSize * pixelSize / 2,
                            padding.Height + (int)(((bounds.Height - 2 * padding.Height) * inputs[i].Count) / (float)(inputs[i + 1].Count + 1) * (j + 1) - inputs[i + 1][j].Height * pixelSize / 2 - ((bounds.Height - 2 * padding.Height) * (inputs[i].Count - 1)) / (float)(2 * inputs[i + 1].Count + 2) + ConvNeuron.filterSize * pixelSize / 2)
                        );
                    }
                    for (int k = 0; k < inputs[i].Count; k++)
                    {
                        g.DrawLine(Pens.Black,
                            padding.Width * (3 + 2 * i) - inputs[i + 1][j * inputs[i + 1].Count / featureMaps[i].Count + k].Width * pixelSize / 2 + currentWidth + ConvNeuron.filterSize * pixelSize + inputs[i + 1][0].Width * pixelSize,
                            padding.Height + (int)((bounds.Height - 2 * padding.Height) / (float)(inputs[i + 1].Count + 1) * (j * inputs[i + 1].Count / featureMaps[i].Count + k + 1) - inputs[i + 1][j * inputs[i + 1].Count / featureMaps[i].Count + k].Height * pixelSize / 2) + inputs[i + 1][j * inputs[i + 1].Count / featureMaps[i].Count + k].Height * pixelSize / 2,
                            padding.Width * (2 + 2 * i) + currentWidth + ConvNeuron.filterSize * pixelSize / 2,
                            padding.Height + (int)(((bounds.Height - 2 * padding.Height) * inputs[i].Count) / (float)(inputs[i + 1].Count + 1) * (j + 1) - inputs[i + 1][j].Height * pixelSize / 2 - ((bounds.Height - 2 * padding.Height) * (inputs[i].Count - 1)) / (float)(2 * inputs[i + 1].Count + 2) + ConvNeuron.filterSize * pixelSize / 2));


                    }
                    g.DrawImage(Scale(featureMaps[i][j].weightsBitmap),
                        padding.Width * (2 + 2 * i) + currentWidth,
                        padding.Height + (int)(((bounds.Height - 2 * padding.Height) * inputs[i].Count) / (float)(inputs[i + 1].Count + 1) * (j + 1) - inputs[i + 1][j].Height * pixelSize / 2 - ((bounds.Height - 2 * padding.Height) * (inputs[i].Count - 1)) / (float)(2 * inputs[i + 1].Count + 2)));
                }
                currentWidth += ConvNeuron.filterSize * pixelSize;
            }
            currentWidth = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                for (int j = 0; j < inputs[i].Count; j++)
                {
                    g.DrawImage(Scale(inputs[i][j]),
                        padding.Width * (1 + 2 * i) + currentWidth,
                        padding.Height + (int)((bounds.Height - 2 * padding.Height) / (float)(inputs[i].Count + 1) * (j + 1) - inputs[i][j].Height * pixelSize / 2));
                }
                currentWidth += inputs[i][0].Width * pixelSize + ConvNeuron.filterSize * pixelSize;
            }
            for (int i = 0; i < fullyConnected.Count; i++)
            {
                for (int j = 0; j < fullyConnected[i].Count; j++)
                {
                    int fullyConnectedThreshold = (int)(fullyConnected[i][j].threshold * 255);
                    g.DrawRectangle(Pens.Black,
                        padding.Width * (3 + 2 * (featureMaps.Count + i)) + currentWidth,
                        padding.Height + (int)(((bounds.Height - 2 * padding.Height) / (float)(fullyConnected[i].Count + 1)) * (j + 1)),
                        pixelSize,
                        pixelSize);
                    g.FillRectangle(new SolidBrush(Color.FromArgb(fullyConnectedThreshold, fullyConnectedThreshold, fullyConnectedThreshold)),
                        padding.Width * (3 + 2 * (featureMaps.Count + i)) + currentWidth,
                        padding.Height + (int)(((bounds.Height - 2 * padding.Height) / (float)(fullyConnected[i].Count + 1)) * (j + 1)),
                        pixelSize,
                        pixelSize);
                }
            }
            for (int i = 0; i < fullyConnected[0].Count; i++)
            {
                for (int j = 0; j < fullyConnected[0][i].weights.Count; j++)
                {
                    int greyscale = (int)(fullyConnected[0][i].weights[j] * 255);
                    g.DrawLine(new Pen(Color.FromArgb(greyscale, greyscale, greyscale)),
                        padding.Width * (3 + 2 * featureMaps.Count) + currentWidth + 3,
                        padding.Height + (int)(((bounds.Height - 2 * padding.Height) / (float)(fullyConnected[0].Count + 1)) * (i + 1)) + 3,
                        padding.Width * (1 + 2 * (inputs.Count - 1)) + currentWidth - (inputs[inputs.Count - 1][0].Width * pixelSize + ConvNeuron.filterSize * pixelSize) + 3,
                        padding.Height + (int)((bounds.Height - 2 * padding.Height) / (float)(inputs[inputs.Count - 1].Count + 1) * (j + 1) - inputs[inputs.Count - 1][j].Height * pixelSize / 2) + 3);
                }
            }
            g.DrawString("Average cost: " + (int)(avgCost * 1000)/1000.0, new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 10);
            g.DrawString("Generation " + Form1.generation, new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 40);
            Form1.generation++;
        }
    }
    public struct ConvNeuron
    {
        public int filterSize;
        public const int subsampleSize = 4;
        public string DNA;
        public float threshold;
        public float[,] weights;
        public const int sectionLength = 4;
        public static readonly int divisor = (int)Math.Pow(10, sectionLength);
        public Bitmap weightsBitmap;
        public ConvNeuron(string gene)
        {
            DNA = gene;
            filterSize = (int)Math.Sqrt(gene.Where(c => c == ' ').Count() - 1);
            weightsBitmap = new Bitmap(filterSize, filterSize);
            weights = new float[filterSize, filterSize];
            threshold = float.Parse(gene.Substring(0, gene.IndexOf(' ')));
            gene = gene.Substring(sectionLength);
            int currentWeightNum = 0;
            while (gene.Length >= sectionLength)
            {
                weights[currentWeightNum / filterSize, currentWeightNum % filterSize] = float.Parse(gene.Substring(0, gene.IndexOf(' '))); ;
                gene = gene.Substring(gene.IndexOf(' ') + 1);
                currentWeightNum++;
            }
            for (currentWeightNum = 0; currentWeightNum < filterSize; currentWeightNum++)
            {
                for (int j = 0; j < filterSize; j++)
                {
                    weightsBitmap.SetPixel(j, currentWeightNum, Color.FromArgb((int)(weights[currentWeightNum, j] * 255), (int)(weights[currentWeightNum, j] * 255), (int)(weights[currentWeightNum, j] * 255)));
                }
            }
        }
        public Bitmap Convolve(Bitmap input)
        {
            Bitmap output = new Bitmap(input);
            for (int i = 0; i < output.Height; i++)
            {
                for (int j = 0; j < output.Width; j++)
                {
                    float outputPixel = 0;
                    for (int k = 0; k < filterSize; k++)
                    {
                        for (int l = 0; l < filterSize; l++)
                        {
                            outputPixel += GetPixel(input, j - filterSize / 2 + l, i - filterSize / 2 + k) / 255.0 * weights[k][l];
                        }
                    }
                    outputPixel /= filterSize * filterSize;
                    outputPixel = 255 * ActivationFunction(outputPixel);
                    output.SetPixel(j, i, Color.FromArgb((int)outputPixel, (int)outputPixel, (int)outputPixel));
                }
            }
            return output;
        }
        public static Bitmap Subsample(Bitmap input)
        {
            Bitmap output = new Bitmap(input, (int)Math.Ceiling(input.Width / (float)subsampleSize), (int)Math.Ceiling(input.Height / (float)subsampleSize));
            List<int> samples = new List<int>();
            for (int i = 0; i < input.Height; i += subsampleSize)
            {
                for (int j = 0; j < input.Width; j += subsampleSize)
                {
                    for (int k = 0; k < subsampleSize; k++)
                    {
                        for (int l = 0; l < subsampleSize; l++)
                        {
                            samples.Add(GetPixel(input, j + l, i + k));
                        }
                    }
                    int max = samples.Max();
                    output.SetPixel(j / subsampleSize, i / subsampleSize, Color.FromArgb(max, max, max));
                }
            }
            return output;
        }
        public static float ActivationFunction(float input)
        {
            return Math.Max(0, input);
        }
        public static int GetPixel(Bitmap b, int x, int y)
        {
            if (x < 0 || x >= b.Width || y < 0 || y >= b.Height)
                return 0;
            return b.GetPixel(x, y).R;
        }
    }
    public class Neuron
    {
        public const float maxWeightLineWidth = 2;
        public const float thresholdEllipseDiameter = 20;
        public const float thresholdEllipseOutlineWidth = 1;
        public string DNA;
        public float errorSignal;
        public List<float[]> currentBatchWeightsNudges;
        public float[] weights;
        public float[] input;
        public float weightedSum;
        public float output;
        public bool finalLayer;
        public Neuron(string gene, bool isFinalLayer = false)
        {
            DNA = gene;
            weights = new float[gene.Where(c => c == ' ').Count()];
            int weight = 0;
            errorSignal = 0;
            currentBatchWeightsNudges = new List<float[]>();
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
        float ActivationFunction(float input)
        {
            return finalLayer ? Sigmoid(input) : SoftPlus(input);
        }
        float ActivationFunctionDerivative(float input)
        {
            return finalLayer ? SigmoidDerivative(input) : SoftPlusDerivative(input);
        }
        public static float LnMirrored(float input)
        {
            return (float)Math.Log(Math.Abs(input) + 1);
        }
        public static float LnMirroredDerivative(float input)
        {
            return (float)(1 / (Math.Abs(input) + 1));
        }
        public static float SigmoidDerivative(float input)
        {
            float ex = Math.Pow(2.718281828459, input);
            return (float)(ex / (1 + ex) / (1 + ex));
        }
        public static float TanhDerivative(float input)
        {
            float tanh = Math.Tanh(input);
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
        public void StoreChanges(float learningRate)
        {
            currentBatchWeightsNudges.Add(new float[weights.Length]);
            for (int i = 0; i < weights.Length; i++)
            {
                currentBatchWeightsNudges[currentBatchWeightsNudges.Count - 1][i] = Bound((input[i] * errorSignal) * learningRate, -1, 1);
            }
        }
        public static float Bound(float f, float min, float max)
        {
            return Math.Max(Math.Min(f, max), min);
        }
        public void MakeChanges(float lambda)
        {
            List<float> weightChanges = new List<float>();
            for (int j = 0; j < currentBatchWeightsNudges[0].Length; j++)
            {
                float total = 0;
                for (int k = 0; k < currentBatchWeightsNudges.Count; k++)
                    total += currentBatchWeightsNudges[k][j];
                total += weights[j] * lambda / currentBatchWeightsNudges.Count;
                weightChanges.Add(total / currentBatchWeightsNudges.Count);
            }
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = Bound(weights[i] - weightChanges[i], -2, 2);
            }
            currentBatchWeightsNudges = new List<float[]>();
        }
    }
    public struct Digit
    {
        public int value;
        public float[] pixelValues;
        public Digit(string path)
        {
            value = int.Parse(Directory.GetParent(path).Name);
            Bitmap image = new Bitmap(path);
            pixelValues = new float[image.Width * image.Height];
            for (int i = 0; i < image.Height; i++)
            {
                for (int j = 0; j < image.Width; j++)
                {
                    pixelValues[i * image.Width + j] = image.GetPixel(j, i).GetBrightness() / 128.0 - 1;
                }
            }
        }
    }
}
