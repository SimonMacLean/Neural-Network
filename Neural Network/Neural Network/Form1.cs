﻿using System;
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
        public const int populationSize = 105;
        public const int tests = 1000;
        private List<NeuralNetwork> population;
        private Timer updatePopulationTimer;
        private string trainingDataPath;
        NeuralNetwork best;
        public Form1()
        {
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
            trainingDataPath = "C:\\Users\\simon\\Documents\\Visual Studio 2015\\Projects\\Neural Network Try 2\\Neural Network Try 2\\Images\\";
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
            foreach (NeuralNetwork n in population)
            {
                grades.Add(n.correct);
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
                if (topScorers.Count >= 15)
                    break;
            }
            for (int i = 0; topScorers.Count < 15; i++)
            {
                if (places.Contains(i))
                    continue;
                topScorers.Add(population[i]);
            }
            topScorers = Randomize(topScorers, R);
            List<NeuralNetwork> newPopulation = new List<NeuralNetwork>();
            for (int i = 14; i > 0; i--)
            {
                for (int j = 0; j < i; j++)
                {
                    newPopulation.Add(new NeuralNetwork(topScorers[14 - i], topScorers[15 - i]));
                }
            }
            population.RemoveAt(0);
            population.Add(best);
            population = newPopulation;
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
        }
        public void DrawGraph(Graphics g, Rectangle Bounds, List<double> scores)
        {
            
        }
    }
    public struct NeuralNetwork
    {
        public const int imageWidth = 28;
        public const int imageHeight = 28;
        public static Random mutationRandom;
        public List<List<Neuron>> neurons;
        public int testsDone;
        public int correct;
        private Bitmap current;
        private List<List<double>> inputs;
        private List<double> outputs;
        private const int mutationFrequency = 15;
        public static bool operator >(NeuralNetwork a, NeuralNetwork b)
        {
            return a.correct > b.correct;
        }
        public static bool operator <(NeuralNetwork a, NeuralNetwork b)
        {
            return a.correct < b.correct;
        }
        public NeuralNetwork(List<int> layersizes)
        {
            mutationRandom = new Random();
            inputs = new List<List<double>>();
            neurons = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            current = new Bitmap(28, 28);
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
            inputs = new List<List<double>>();
            neurons = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            current = new Bitmap(28, 28);
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
            inputs = new List<List<double>>();
            neurons = new List<List<Neuron>>();
            testsDone = 0;
            correct = 0;
            current = new Bitmap(28, 28);
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
            testsDone = original.testsDone;
            correct = original.correct;
            current = new Bitmap(28, 28);
            outputs = original.outputs;
        }
        public List<double> getOutputs(Bitmap input)
        {
            inputs = new List<List<double>>();
            inputs.Add(new List<double>());
            for (int i = 0; i < imageHeight; i++)
            {
                for (int j = 0; j < imageWidth; j++)
                {
                    Color c = input.GetPixel(j, i);
                    double brightness = ((c.R + c.G + c.B) / 3.0) / 255.0;
                    inputs[0].Add(brightness);
                }
            }
            for (int i = 0; i < neurons.Count; i++)
            {
                inputs.Add(new List<double>());
                for (int j = 0; j < neurons[i].Count; j++)
                {
                    inputs[i + 1].Add(neurons[i][j].GetOutput(inputs[i].ToArray()));
                }
            }
            return inputs[inputs.Count - 1];
        }
        public int getLikeliestOutput(Bitmap input)
        {
            List<double> outputs = getOutputs(input);
            return outputs.IndexOf(outputs.Max());
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
            g.DrawString("" + (int)(correct / (double)testsDone * 1000) / 10.0 + "% accuracy", new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 10);
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
