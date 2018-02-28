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

namespace Neural_Network
{
    public partial class Form1 : Form
    {
        public static int generation = 1;
        private static Random R;
        const int bestPopulation = 5;
        public int populationSize;
        public const int tests = 100;
        private List<NeuralNetwork> population;
        private Timer updatePopulationTimer;
        private string trainingDataPath;
        NeuralNetwork best;
        List<double> scores = new List<double>();
        public Form1()
        {
            populationSize = 0;
            for(int i = 0; i < bestPopulation; i++)
            {
                populationSize += i + 1;
            }
            InitializeComponent();
            R = new Random();
            population = new List<NeuralNetwork>();
            for (int i = 0; i < populationSize; i++)
            {
                population.Add(new NeuralNetwork(16, 16, 10));
            }
            updatePopulationTimer = new Timer();
            updatePopulationTimer.Enabled = true;
            updatePopulationTimer.Interval = 1;
            updatePopulationTimer.Tick += UpdatePopulation;
            trainingDataPath = Path.Combine(Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, @"..\..\..")), @"Images\");
            scores.Add(0);
        }

        private void UpdatePopulation(object sender, EventArgs e)
        {
            for (int i = 0; i < tests; i++)
            {
                string[] files = Directory.GetFiles(trainingDataPath, "*.bmp", SearchOption.AllDirectories);
                int number = R.Next(files.Length);
                string randomTrainingDataPath = files[number];
                Bitmap randomTrainingData = new Bitmap(randomTrainingDataPath);
                for (int j = 0; j < population.Count; j++)
                {
                    NeuralNetwork nn = population[j];
                    List<double> outputs = nn.getOutputs(randomTrainingData).ToList();
                    nn.avgCost = (nn.avgCost * nn.testsDone + nn.getCost(int.Parse(Directory.GetParent(randomTrainingDataPath).Name), outputs)) / (nn.testsDone + 1);
                    nn.testsDone++;
                    if (nn.mostConfidentOutput(outputs) == int.Parse(Directory.GetParent(randomTrainingDataPath).Name))
                    {
                        nn.correct++;
                    }
                    population[j] = nn;
                }
            }
            List<double> grades = new List<double>();
            foreach (NeuralNetwork n in population)
            {
                grades.Add(n.avgCost);
            }
            NeuralNetwork[] arrayVersion = population.ToArray();
            Array.Sort(grades.ToArray(), arrayVersion);
            population = arrayVersion.Reverse().ToList();
            best = new NeuralNetwork(population[0]);
            List<NeuralNetwork> topScorers = new List<NeuralNetwork>();
            List<int> places = new List<int>();
            for (int i = 0; i < population.Count; i++)
            {
                if (R.Next(population.Count) > i)
                {
                    topScorers.Add(population[i]);
                    places.Add(i);
                }
                if (topScorers.Count > bestPopulation)
                    break;
            }
            for (int i = 0; topScorers.Count <= bestPopulation; i++)
            {
                if (places.Contains(i))
                    continue;
                topScorers.Add(population[i]);
            }
            topScorers = Randomize(topScorers, R);
            List<NeuralNetwork> newPopulation = new List<NeuralNetwork>();
            for (int i = bestPopulation; i > 0; i--)
            {
                for (int j = 0; j < i; j++)
                {
                    newPopulation.Add(new NeuralNetwork(topScorers[bestPopulation - i], topScorers[bestPopulation + 1 - i]));
                }
            }
            population.RemoveAt(0);
            population.Add(best);
            population = newPopulation;
            scores.Add(((double)best.correct / best.testsDone) * 100);
            Invalidate();
        }
        List<NeuralNetwork> Randomize(List<NeuralNetwork> sorted, Random r)
        {
            List<NeuralNetwork> result = new List<NeuralNetwork>();
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
            if(scores.Count > 1)
                DrawGraph(e.Graphics, new Rectangle(10, 90, 210, 140), scores);
        }
        public void DrawGraph(Graphics g, Rectangle Bounds, List<double> scores)
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
    public struct NeuralNetwork
    {
        public double avgCost;
        public const int imageWidth = 28;
        public const int imageHeight = 28;
        public static Random mutationRandom;
        public List<List<Neuron>> neurons;
        public int testsDone;
        public int correct;
        private Bitmap current;
        private List<double[]> inputs;
        private List<double> outputs;
        private const int mutationFrequency = 15;
        public static bool operator >(NeuralNetwork a, NeuralNetwork b)
        {
            return a.avgCost > b.avgCost;
        }
        public static bool operator <(NeuralNetwork a, NeuralNetwork b)
        {
            return a.avgCost < b.avgCost;
        }
        public NeuralNetwork(List<int> layersizes)
        {
            mutationRandom = new Random();
            inputs = new List<double[]>();
            neurons = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            avgCost = 0;
            current = new Bitmap(imageWidth, imageHeight);
            outputs = new List<double> { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            int layerSize = imageWidth * imageHeight;
            for (int i = 0; i < layersizes.Count; i++)
            {
                neurons.Add(new List<Neuron>());
                int prevLayerSize = layerSize;
                layerSize = layersizes[i];
                for (int j = 0; j < layerSize; j++)
                {
                    string neuronString = string.Empty;
                    for (int k = 0; k < prevLayerSize; k++)
                    {
                        neuronString += mutationRandom.Next(Neuron.divisor + 1);
                    }
                    neurons[i].Add(new Neuron(neuronString));
                }
            }
        }
        public NeuralNetwork(params int[] layersizes)
        {
            mutationRandom = new Random();
            inputs = new List<double[]>();
            neurons = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            avgCost = 0;
            current = new Bitmap(imageWidth, imageHeight);
            outputs = new List<double> { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            int layerSize = imageWidth * imageHeight;
            for (int i = 0; i < layersizes.Length; i++)
            {
                neurons.Add(new List<Neuron>());
                int prevLayerSize = layerSize;
                layerSize = layersizes[i];
                for (int j = 0; j < layerSize; j++)
                {
                    string neuronString = string.Empty;
                    neuronString += (mutationRandom.Next(Neuron.divisor) + Neuron.divisor).ToString().Substring(1);
                    for (int k = 0; k < prevLayerSize; k++)
                    {
                        neuronString += (mutationRandom.Next(Neuron.divisor) + Neuron.divisor).ToString().Substring(1);
                    }
                    neurons[i].Add(new Neuron(neuronString));
                }
            }
        }
        public NeuralNetwork(NeuralNetwork parentA, NeuralNetwork parentB)
        {
            mutationRandom = new Random();
            inputs = new List<double[]>();
            neurons = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            avgCost = 0;
            current = new Bitmap(imageWidth, imageHeight);
            outputs = new List<double> { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            int layerSize = imageWidth * imageHeight;
            for (int i = 0; i < parentA.neurons.Count; i++)
            {
                neurons.Add(new List<Neuron>());
                layerSize = parentA.neurons[i].Count;
                for (int j = 0; j < layerSize; j++)
                {
                    string neuronADNA = parentA.neurons[i][j].DNA;
                    string neuronBDNA = parentB.neurons[i][j].DNA;
                    string crossoveredDNA = string.Empty;
                    for (int k = 0; k < neuronADNA.Length; k++)
                    {
                        crossoveredDNA += (mutationRandom.Next(2) == 0 ? neuronADNA : neuronBDNA)[k];
                        if (mutationRandom.Next(100 / mutationFrequency) == 0)
                            crossoveredDNA = crossoveredDNA.Substring(0, k) + mutationRandom.Next(10);
                    }
                    neurons[i].Add(new Neuron(crossoveredDNA));
                }
            }
        }
        public NeuralNetwork(NeuralNetwork original)
        {
            mutationRandom = new Random();
            inputs = original.inputs;
            neurons = original.neurons;
            testsDone = 0;
            correct = 0;
            avgCost = 0;
            current = new Bitmap(imageWidth, imageHeight);
            outputs = original.outputs;
        }
        public double[] getOutputs(Bitmap input)
        {
            inputs = new List<double[]>();
            inputs.Add(new double[784]);
            for(int i = 0; i < imageHeight; i++)
            {
                for(int j = 0; j < imageHeight; j++)
                {
                    inputs[0][i * imageWidth + j] = input.GetPixel(j, i).GetBrightness() / 256.0;
                }
            }
            for (int i = 0; i < neurons.Count; i++)
            {
                inputs.Add(new double[neurons[i].Count]);
                for (int j = 0; j < neurons[i].Count; j++)
                {
                    inputs[i + 1][j] = neurons[i][j].GetOutput(inputs[i].ToArray());
                }
            }
            return inputs[inputs.Count - 1];
        }
        public int mostConfidentOutput(List<double> outputs)
        {
            return outputs.ToList().IndexOf(outputs.Max());
        }
        public double getCost(int desired, List<double> output)
        {
            double totalCost = 0;
            for(int i = 0; i < output.Count; i++)
            {
                if(i == desired)
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
        public void Draw(Graphics g, Rectangle bounds)
        {
            if (neurons == null)
                return;
            Size padding = new Size(bounds.Width / (2 * neurons.Count), bounds.Height / (2 * neurons[0].Count));
            for (int i = 0; i < neurons.Count; i++)
            {
                for (int j = 0; j < neurons[i].Count; j++)
                {
                    if (i > 0)
                    {
                        for (int k = 0; k < neurons[i - 1].Count; k++)
                        {
                            g.DrawLine(new Pen(Color.FromArgb((int)(neurons[i][j].weights[k] * 255), (int)(neurons[i][j].weights[k] * 255), (int)(neurons[i][j].weights[k] * 255))),
                                (int)(padding.Width * (1 + 2.0 * i)), (int)(padding.Height + (bounds.Height - 2.0 * padding.Height) / (neurons[i].Count - 1) * j),
                                (int)(padding.Width * (2.0 * i - 1)), (int)(padding.Height + (bounds.Height - 2.0 * padding.Height) / (neurons[i - 1].Count - 1) * k));
                        }
                    }
                    g.FillEllipse(new SolidBrush(Color.FromArgb((int)(neurons[i][j].threshold * 255), (int)(neurons[i][j].threshold * 255), (int)(neurons[i][j].threshold * 255))),
                        (int)(padding.Width * (1 + 2.0 * i)) - 5, (int)(padding.Height + (bounds.Height - 2.0 * padding.Height) / (neurons[i].Count - 1) * j) - 5,
                        10, 10);
                }
            }
            g.DrawString("" + (int)(correct / (double)testsDone * 1000) / 10.0 + "% correct", new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 10);
            g.DrawString("Generation " + Form1.generation, new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 40);
            Form1.generation++;
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
            return Math.Max(0, total);
        }
    }
}
