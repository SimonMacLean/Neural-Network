using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Covolutional_Neural_Network
{
    public partial class Form1 : Form
    {
        public static int generation = 1;
        private static Random R;
        public const int populationSize = 15;
        public const int tests = 10;
        private List<ConvNet> population;
        private Timer updatePopulationTimer;
        private string trainingDataPath;
        ConvNet best;
        public Form1()
        {
            InitializeComponent();
            R = new Random();
            population = new List<ConvNet>();
            for (int i = 0; i < populationSize; i++)
            {
                population.Add(new ConvNet(6, 3, 3, 37, 10));
            }
            updatePopulationTimer = new Timer();
            updatePopulationTimer.Enabled = true;
            updatePopulationTimer.Interval = 1;
            updatePopulationTimer.Tick += UpdatePopulation;
            trainingDataPath = @"E:\Neural Network\Images\";
        }

        private void UpdatePopulation(object sender, EventArgs e)
        {
            string[] files = Directory.GetFiles(trainingDataPath, "*.bmp", SearchOption.AllDirectories);
            for (int i = 0; i < tests; i++)
            {
                int number = R.Next(files.Length);
                string randomTrainingDataPath = files[number];
                Bitmap randomTrainingData = new Bitmap(randomTrainingDataPath);
                for (int j = 0; j < population.Count; j++)
                {
                    ConvNet nn = population[j];
                    int guess = nn.getLikeliestOutput(randomTrainingData);
                    nn.testsDone++;
                    if (guess == int.Parse(Directory.GetParent(randomTrainingDataPath).Name))
                    {
                        nn.correct++;
                    }
                    population[j] = nn;
                }
            }
            List<int> grades = new List<int>();
            foreach (ConvNet n in population)
            {
                grades.Add(n.correct);
            }
            ConvNet[] arrayVersion = population.ToArray();
            Array.Sort(grades.ToArray(), arrayVersion);
            population = arrayVersion.Reverse().ToList();
            best = new ConvNet(population[0]);
            List<ConvNet> topScorers = new List<ConvNet>();
            List<int> places = new List<int>();
            for (int i = 0; i < population.Count; i++)
            {
                if (R.Next(population.Count) > i)
                {
                    topScorers.Add(population[i]);
                    places.Add(i);
                }
                if (topScorers.Count >= 5)
                    break;
            }
            for (int i = 0; topScorers.Count < 5; i++)
            {
                if (places.Contains(i))
                    continue;
                topScorers.Add(population[i]);
            }
            topScorers = Randomize(topScorers, R);
            List<ConvNet> newPopulation = new List<ConvNet>();
            for (int i = 4; i > 0; i--)
            {
                for (int j = 0; j < i; j++)
                {
                    newPopulation.Add(new ConvNet(topScorers[4 - i], topScorers[5 - i]));
                }
            }
            population.RemoveAt(0);
            population.Add(best);
            population = newPopulation;
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

        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            best.Draw(e.Graphics, ClientRectangle);
        }
    }
    public struct Neuron
    {
        public string DNA;
        public double threshold;
        public List<double> weights;
        public const int sectionLength = 4;
        public static readonly int divisor = (int)Math.Pow(10, sectionLength);
        public Neuron(string gene)
        {
            DNA = gene;
            weights = new List<double>();
            threshold = int.Parse(gene.Substring(0, sectionLength)) / (double)divisor;
            gene = gene.Substring(sectionLength);
            while (gene.Length >= sectionLength)
            {
                weights.Add(int.Parse(gene.Substring(0, sectionLength)) / (double)divisor);
                gene = gene.Substring(sectionLength);
            }
        }
        public double GetOutput(double[] input)
        {
            double total = 0;
            for (int i = 0; i < input.Length; i++)
            {
                total += input[i] * weights[i];
            }
            total /= input.Length;
            //return 1 / (1 + 1 / (Math.Pow(Math.E, Math.PI * (total - threshold))));
            //return total >= threshold ? 1 : 0;
            return ConvNeuron.ActivationFunction(total - threshold);
        }
    }
    public struct ConvNet
    {
        public const int imageWidth = 28;
        public const int imageHeight = 28;
        public const int fullyConnectedLayers = 2;
        public const int pixelSize = 5;
        public static Random mutationRandom;
        public List<List<ConvNeuron>> featureMaps;
        public List<List<Neuron>> fullyConnected;
        public int testsDone;
        public int correct;
        private Bitmap current;
        private List<List<Bitmap>> inputs;
        private List<List<double>> fullyConnectedInputs;
        private List<double> outputs;
        private const int mutationFrequency = 15;
        public static bool operator >(ConvNet a, ConvNet b)
        {
            return a.correct > b.correct;
        }
        public static bool operator <(ConvNet a, ConvNet b)
        {
            return a.correct < b.correct;
        }
        public ConvNet(List<int> layerSizes)
        {
            mutationRandom = new Random();
            inputs = new List<List<Bitmap>>();
            fullyConnectedInputs = new List<List<double>>();
            featureMaps = new List<List<ConvNeuron>>();
            fullyConnected = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            current = new Bitmap(28, 28);
            outputs = new List<double>();
            for (int i = 0; i < layerSizes[layerSizes.Count - 1]; i++)
            {
                outputs.Add(0);
            }
            int prevLayerSize = 1;
            for (int i = 0; i < layerSizes.Count - fullyConnectedLayers; i++)
            {
                featureMaps.Add(new List<ConvNeuron>());
                int layerSize = layerSizes[i];
                prevLayerSize *= layerSize;
                for (int j = 0; j < layerSize; j++)
                {
                    string neuronString = mutationRandom.Next(Neuron.divisor).ToString();
                    for (int k = 0; k < ConvNeuron.filterSize * ConvNeuron.filterSize + 1; k++)
                    {
                        neuronString += mutationRandom.Next(Neuron.divisor);
                    }
                    featureMaps[i].Add(new ConvNeuron(neuronString));
                }
            }
            for (int i = 0; i < fullyConnectedLayers; i++)
            {
                fullyConnected.Add(new List<Neuron>());
                for (int j = 0; j < layerSizes[layerSizes.Count - fullyConnectedLayers + i]; j++)
                {
                    string neuronString = mutationRandom.Next(Neuron.divisor).ToString();
                    for (int k = 0; k < prevLayerSize; k++)
                    {
                        neuronString += mutationRandom.Next(Neuron.divisor);
                    }
                    fullyConnected[i].Add(new Neuron(neuronString));
                }
                prevLayerSize = fullyConnected[i].Count;
            }
        }
        public ConvNet(params int[] layerSizes)
        {
            mutationRandom = new Random();
            inputs = new List<List<Bitmap>>();
            fullyConnectedInputs = new List<List<double>>();
            featureMaps = new List<List<ConvNeuron>>();
            fullyConnected = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            current = new Bitmap(28, 28);
            outputs = new List<double>();
            for (int i = 0; i < layerSizes[layerSizes.Length - 1]; i++)
            {
                outputs.Add(0);
            }
            int prevLayerSize = 1;
            for (int i = 0; i < layerSizes.Length - fullyConnectedLayers; i++)
            {
                featureMaps.Add(new List<ConvNeuron>());
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
            fullyConnectedInputs = new List<List<double>>();
            featureMaps = new List<List<ConvNeuron>>();
            fullyConnected = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            current = new Bitmap(28, 28);
            outputs = new List<double>();
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
            testsDone = original.testsDone;
            correct = original.correct;
            current = new Bitmap(28, 28);
            outputs = original.outputs;
        }
        public List<double> getOutputs(Bitmap input)
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
            fullyConnectedInputs = new List<List<double>>();
            fullyConnectedInputs.Add(new List<double>());
            foreach (Bitmap b in inputs[inputs.Count - 1])
            {
                fullyConnectedInputs[0].Add(b.GetPixel(0, 0).R / 255.0);
            }
            for (int i = 0; i < fullyConnectedLayers; i++)
            {
                fullyConnectedInputs.Add(new List<double>());
                for (int j = 0; j < fullyConnected[i].Count; j++)
                {
                    fullyConnectedInputs[i + 1].Add(fullyConnected[i][j].GetOutput(fullyConnectedInputs[i].ToArray()));
                }
            }
            return fullyConnectedInputs[inputs.Count - 2];
        }
        public int getLikeliestOutput(Bitmap input)
        {
            List<double> outputs = getOutputs(input);
            return outputs.IndexOf(outputs.Max());
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
                            padding.Height + (int)((bounds.Height - 2 * padding.Height) / (double)(inputs[i].Count + 1) * (k + 1) - inputs[i][k].Height * pixelSize / 2) + inputs[i][k].Height * pixelSize / 2,
                            padding.Width * (2 + 2 * i) + currentWidth + ConvNeuron.filterSize * pixelSize / 2,
                            padding.Height + (int)(((bounds.Height - 2 * padding.Height) * inputs[i].Count) / (double)(inputs[i + 1].Count + 1) * (j + 1) - inputs[i + 1][j].Height * pixelSize / 2 - ((bounds.Height - 2 * padding.Height) * (inputs[i].Count - 1)) / (double)(2 * inputs[i + 1].Count + 2) + ConvNeuron.filterSize * pixelSize / 2)
                        );
                    }
                    for (int k = 0; k < inputs[i].Count; k++)
                    {
                        g.DrawLine(Pens.Black,
                            padding.Width * (3 + 2 * i) - inputs[i + 1][j * inputs[i + 1].Count / featureMaps[i].Count + k].Width * pixelSize / 2 + currentWidth + ConvNeuron.filterSize * pixelSize + inputs[i + 1][0].Width * pixelSize,
                            padding.Height + (int)((bounds.Height - 2 * padding.Height) / (double)(inputs[i + 1].Count + 1) * (j * inputs[i + 1].Count / featureMaps[i].Count + k + 1) - inputs[i + 1][j * inputs[i + 1].Count / featureMaps[i].Count + k].Height * pixelSize / 2) + inputs[i + 1][j * inputs[i + 1].Count / featureMaps[i].Count + k].Height * pixelSize / 2,
                            padding.Width * (2 + 2 * i) + currentWidth + ConvNeuron.filterSize * pixelSize / 2,
                            padding.Height + (int)(((bounds.Height - 2 * padding.Height) * inputs[i].Count) / (double)(inputs[i + 1].Count + 1) * (j + 1) - inputs[i + 1][j].Height * pixelSize / 2 - ((bounds.Height - 2 * padding.Height) * (inputs[i].Count - 1)) / (double)(2 * inputs[i + 1].Count + 2) + ConvNeuron.filterSize * pixelSize / 2));


                    }
                    g.DrawImage(Scale(featureMaps[i][j].weightsBitmap),
                        padding.Width * (2 + 2 * i) + currentWidth,
                        padding.Height + (int)(((bounds.Height - 2 * padding.Height) * inputs[i].Count) / (double)(inputs[i + 1].Count + 1) * (j + 1) - inputs[i + 1][j].Height * pixelSize / 2 - ((bounds.Height - 2 * padding.Height) * (inputs[i].Count - 1)) / (double)(2 * inputs[i + 1].Count + 2)));
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
                        padding.Height + (int)((bounds.Height - 2 * padding.Height) / (double)(inputs[i].Count + 1) * (j + 1) - inputs[i][j].Height * pixelSize / 2));
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
                        padding.Height + (int)(((bounds.Height - 2 * padding.Height) / (double)(fullyConnected[i].Count + 1)) * (j + 1)),
                        pixelSize,
                        pixelSize);
                    g.FillRectangle(new SolidBrush(Color.FromArgb(fullyConnectedThreshold, fullyConnectedThreshold, fullyConnectedThreshold)),
                        padding.Width * (3 + 2 * (featureMaps.Count + i)) + currentWidth,
                        padding.Height + (int)(((bounds.Height - 2 * padding.Height) / (double)(fullyConnected[i].Count + 1)) * (j + 1)),
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
                        padding.Height + (int)(((bounds.Height - 2 * padding.Height) / (double)(fullyConnected[0].Count + 1)) * (i + 1)) + 3,
                        padding.Width * (1 + 2 * (inputs.Count - 1)) + currentWidth - (inputs[inputs.Count - 1][0].Width * pixelSize + ConvNeuron.filterSize * pixelSize) + 3,
                        padding.Height + (int)((bounds.Height - 2 * padding.Height) / (double)(inputs[inputs.Count - 1].Count + 1) * (j + 1) - inputs[inputs.Count - 1][j].Height * pixelSize / 2) + 3);
                }
            }
            g.DrawString("" + (int)(correct / (double)testsDone * 1000) / 10.0 + "% accuracy", new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 10);
            g.DrawString("Generation " + Form1.generation, new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 40);
            Form1.generation++;
        }
    }
    public struct ConvNeuron
    {
        public const int filterSize = 5;
        public const int subsampleSize = 4;
        public string DNA;
        public double threshold;
        public double[][] weights;
        public const int sectionLength = 4;
        public static readonly int divisor = (int)Math.Pow(10, sectionLength);
        public Bitmap weightsBitmap;
        public ConvNeuron(string gene)
        {
            weightsBitmap = new Bitmap(filterSize, filterSize);
            DNA = gene;
            weights = new double[filterSize][];
            threshold = int.Parse(gene.Substring(0, sectionLength)) / (double)divisor;
            gene = gene.Substring(sectionLength);
            int i = 0;
            while (gene.Length >= sectionLength)
            {
                if (i % filterSize == 0)
                {
                    weights[i / filterSize] = new double[filterSize];
                }
                weights[i / filterSize][i % filterSize] = int.Parse(gene.Substring(0, sectionLength)) / (double)divisor;
                gene = gene.Substring(sectionLength);
                i++;
            }
            for (i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    weightsBitmap.SetPixel(j, i, Color.FromArgb((int)(weights[i][j] * 255), (int)(weights[i][j] * 255), (int)(weights[i][j] * 255)));
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
                    double outputPixel = 0;
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
            Bitmap output = new Bitmap(input, (int)Math.Ceiling(input.Width / (double)subsampleSize), (int)Math.Ceiling(input.Height / (double)subsampleSize));
            for (int i = 0; i < input.Height; i += subsampleSize)
            {
                for (int j = 0; j < input.Width; j += subsampleSize)
                {
                    List<int> samples = new List<int>();
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
        public static double ActivationFunction(double input)
        {
            return Math.Max(0, input);
        }
        public static int GetPixel(Bitmap b, int x, int y)
        {
            if (x < 0 || x >= b.Width || y < 0 || y >= b.Height)
            {
                return 0;
            }
            return b.GetPixel(x, y).R;
        }
    }
}
