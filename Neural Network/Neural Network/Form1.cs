/*public partial class Form1 : Form
{
    public static int generation = 1;
    private static Random R;
    const int bestPopulation = 5;
    public int populationSize;
    public const int tests = 100;
    private List<ConvNet> population;
    private Timer updatePopulationTimer;
    private string trainingDataPath;
    ConvNet best;
    public Form1()
    {
        InitializeComponent();
        R = new Random();
        population = new List<ConvNet>();
        populationSize = bestPopulation * (bestPopulation + 1) / 2;
        for (int i = 0; i < populationSize; i++)
        {
            population.Add(new ConvNet(6, 3, 3, 37, 10));
        }
        updatePopulationTimer = new Timer();
        updatePopulationTimer.Enabled = true;
        updatePopulationTimer.Interval = 1;
        updatePopulationTimer.Tick += UpdatePopulation;
        trainingDataPath = Path.Combine(Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, @"..\..\..")), @"Images\");
    }

    private void UpdatePopulation(object sender, EventArgs e)
    {
        string[] files = Directory.GetFiles(trainingDataPath, "*.bmp", SearchOption.AllDirectories);
        Parallel.For(0, tests, i =>
        {
            int number = R.Next(files.Length);
            string randomTrainingDataPath = files[number];
            Bitmap randomTrainingData = new Bitmap(randomTrainingDataPath);
            for (int j = 0; j < population.Count; j++)
            {
                ConvNet nn = population[j];
                List<double> outputs = nn.getOutputs(randomTrainingData).ToList();
                nn.avgCost = (nn.avgCost * nn.testsDone + nn.getCost(int.Parse(Directory.GetParent(randomTrainingDataPath).Name), outputs)) / (nn.testsDone + 1);
                nn.testsDone++;
                if (nn.mostConfidentOutput(outputs) == int.Parse(Directory.GetParent(randomTrainingDataPath).Name))
                {
                    nn.correct++;
                }
                population[j] = nn;
            }
        });
        List<double> grades = new List<double>();
        foreach (ConvNet n in population)
        {
            grades.Add(n.avgCost);
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
            if (topScorers.Count >= bestPopulation)
                break;
        }
        for (int i = 0; topScorers.Count < bestPopulation; i++)
        {
            if (places.Contains(i))
                continue;
            topScorers.Add(population[i]);
        }
        topScorers = Randomize(topScorers, R);
        List<ConvNet> newPopulation = new List<ConvNet>();
        for (int i = bestPopulation - 1; i > 0; i--)
        {
            for (int j = 0; j < i; j++)
            {
                newPopulation.Add(new ConvNet(topScorers[(bestPopulation - 1) - i], topScorers[bestPopulation - i]));
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
        private Digit[] allDigits;
        public Form1()
        {
            populationSize = 0;
            for(int i = 0; i < bestPopulation; i++)
            {
                populationSize += i + 1;
            }
            InitializeComponent();
            typeof(Form1).InvokeMember("DoubleBuffered",
                BindingFlags.SetProperty | BindingFlags.Instance | BindingFlags.NonPublic, null, this,
                new object[] { true });
            string[] files = Directory.GetFiles(Path.Combine(Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, @"..\..\..")), @"Images\"), "*.bmp", SearchOption.AllDirectories);
            allDigits = new Digit[files.Length];
            for (int i = 0; i < files.Length; i++)
            {
                allDigits[i] = new Digit(files[i]);
            }
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
    }*/
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
using Alea;
using Alea.Parallel;
using System.Reflection;

namespace Neural_Network
{
    public partial class Form1 : Form
    {
        public static int generation = 0;
        public static Random R;
        //const int bestPopulation = 6;
        private const int populationSize = 100;
        private const int tests = 1000;
        private NeuralNetwork[] population;
        private Timer updatePopulationTimer;
        //private string trainingDataPath;
        NeuralNetwork best;
        List<double> scores = new List<double>();
        private Gpu gpu;
        //private string[] files;
        private double[][][][] allNeuronWeights;
        private Neuron[][][] allNeurons;
        private double[][][] allInputs;
        private int[] layersizes;
        private Digit[] allDigits;
        public Form1()
        {
            InitializeComponent();
            gpu = Gpu.Default;
            typeof(Form1).InvokeMember("DoubleBuffered",
                BindingFlags.SetProperty | BindingFlags.Instance | BindingFlags.NonPublic, null, this,
                new object[] { true });
            string[] files = Directory.GetFiles(Path.Combine(Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, @"..\..\..")), @"Images\"), "*.bmp", SearchOption.AllDirectories);
            allDigits = new Digit[files.Length];
            for (int i = 0; i < files.Length; i++)
            {
                allDigits[i] = new Digit(files[i]);
            }
            layersizes = new[] { 137, 57, 24, 10 };
            allNeuronWeights = new double[populationSize][][][];
            allNeurons = new Neuron[populationSize][][];
            allInputs = new double[populationSize][][];
            for (int i = 0; i < populationSize; i++)
            {
                allNeuronWeights[i] = new double[layersizes.Length][][];
                for (int j = 0; j < layersizes.Length; j++)
                {
                    allNeuronWeights[i][j] = new double[layersizes[j]][];
                    allNeuronWeights[i][j][0] = new double[784];
                    for(int k = 1; k < layersizes[j]; k++)
                    {
                        allNeuronWeights[i][j][k] = new double[layersizes[k - 1]];
                    }
                }
            }
            R = new Random();
            population = new NeuralNetwork[populationSize];
            for (int i = 0; i < populationSize; i++)
            {
                population[i] = new NeuralNetwork(i, allNeuronWeights, allNeurons, allInputs);
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
            double[][] inputs = new double[tests][];
            string[] paths = new string[tests];
            double[][] outputs = new double[populationSize * tests][];
            NeuralNetwork[] functionPopulation = population.Select(n => n).ToArray();
            double[][][][] functionNeuronWeights = allNeuronWeights.Select(n => n).ToArray();
            Neuron[][][] functionNeurons = allNeurons.Select(n => n).ToArray();
            double[][][] functionInputs = allInputs.Select(n => n).ToArray();
            int[] values = new int[tests];
            for (int i = 0; i < tests; i++)
            {
                int place = R.Next(allDigits.Length);
                values[i] = allDigits[place].value;
                inputs[i] = allDigits[place].pixelValues;
                for (int j = 0; j < populationSize; j++)
                {
                    outputs[i * populationSize + j] = new double[10];
                }
            }
            Gpu.Default.For(0, population.Length * tests, i =>
            {
                outputs[i] = functionPopulation[i / tests].GetOutputs(inputs[i % tests], functionNeuronWeights, functionNeurons, functionInputs);
            }
            );
            for (int i = 0; i < populationSize * tests; i++)
            {
                NeuralNetwork nn = functionPopulation[i / tests];
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
            double[][][][] newNeuronWeights = new double[allNeuronWeights.Length][][][];
            Neuron[][][] newNeurons = new Neuron[allNeurons.Length][][];
            double[][][] newInputs = new double[allInputs.Length][][];
            List<double> grades = new List<double>();
            foreach (NeuralNetwork n in population)
            {
                grades.Add(n.sumCost);
            }
            Array.Sort(grades.ToArray(), population);
            //population = population.Reverse().ToArray();
            List<NeuralNetwork> topScorers = new List<NeuralNetwork>();
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
            List<NeuralNetwork> newPopulation = new List<NeuralNetwork>();
            for (int i = 0; newPopulation.Count < populationSize; i++)
            {
                for (int j = 0; j < 7 && newPopulation.Count < populationSize; j++)
                {
                    if (j >= 2)
                        if (R.Next(order[i]) == 0)
                            newPopulation.Add(new NeuralNetwork(topScorers[order[i]], topScorers[order[(i + j) % order.Count]], newPopulation.Count, newNeuronWeights, newNeurons, newInputs, allNeuronWeights, allNeurons, allInputs));
                        else
                            newPopulation.Add(new NeuralNetwork(topScorers[order[i]], topScorers[order[(i + j) % order.Count]], newPopulation.Count, newNeuronWeights, newNeurons, newInputs, allNeuronWeights, allNeurons, allInputs));
                }
            }
            best = new NeuralNetwork(population[0], 0, newNeuronWeights, newNeurons, newInputs, allNeuronWeights, allNeurons, allInputs);
            newPopulation[0] = best;
            population = newPopulation.ToArray();
            Array.Copy(newNeuronWeights, allNeuronWeights, 0);
            Array.Copy(newNeurons, allNeurons, 0);
            Array.Copy(newInputs, allInputs, 0);
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
        public const int imageWidth = 28;
        public const int imageHeight = 28;

        //public static Random mutationRandom;

        public double sumCost;
        public int testsDone;
        public int correct;
        int rowIndex;
//        public string guesses;
//        public string actual;
        public NeuralNetwork(int rowIndex, double[][][][] allWeights, Neuron[][][] allNeurons, double[][][] allInputs) : this()
        {
            this.rowIndex = rowIndex;
//            guesses = "";
//            actual = "";
            //mutationRandom = new Random();
            allInputs[rowIndex] = new double[allWeights[rowIndex].Length + 1][];
            allNeurons[rowIndex] = new Neuron[allWeights[rowIndex].Length][];
            testsDone = 0;
            correct = 0;
            sumCost = 0;
            int layerSize = imageWidth * imageHeight;
            allInputs[rowIndex][0] = new double[layerSize];
            for (int i = 0; i < allWeights[rowIndex].Length; i++)
            {
                allNeurons[rowIndex][i] = new Neuron[allWeights[rowIndex][i].GetLength(0)];
                int prevLayerSize = layerSize;
                layerSize = allWeights[rowIndex][i].GetLength(0);
                allInputs[rowIndex][i + 1] = new double[layerSize];
                for (int j = 0; j < layerSize; j++)
                {
                    double threshold = Form1.R.Next(-1 * Neuron.divisor, Neuron.divisor) / (double)Neuron.divisor;
                    double[] weights = new double[prevLayerSize];
                    for (int k = 0; k < prevLayerSize; k++)
                    {
                        weights[k] = Form1.R.Next(-1 * Neuron.divisor, Neuron.divisor) / (double)Neuron.divisor;
                    }
                    allNeurons[rowIndex][i][j] = new Neuron(threshold, j, allWeights[rowIndex][i], weights);
                }
            }
        }
        public NeuralNetwork(NeuralNetwork parentA, NeuralNetwork parentB, int rowIndex, double[][][][] newWeights, Neuron[][][] newNeurons, double[][][] newInputs, double[][][][] oldWeights, Neuron[][][] oldNeurons, double[][][] oldInputs)
        {
            this.rowIndex = rowIndex;
//            guesses = "";
//            actual = "";
            //mutationRandom = new Random();
            int layers = oldNeurons[parentA.rowIndex].Length;
            newInputs[rowIndex] = oldInputs[parentA.rowIndex];
            newNeurons[rowIndex] = oldNeurons[parentA.rowIndex];
            newWeights[rowIndex] = oldWeights[parentA.rowIndex];
            testsDone = 0;
            correct = 0;
            sumCost = 0;
            int layerSize;
            int prevLayerSize = imageWidth * imageHeight;
            for (int i = 0; i < layers; i++)
            {
                layerSize = oldNeurons[parentA.rowIndex][i].Length;
                for (int j = 0; j < layerSize; j++)
                {
                    Neuron neuronA = oldNeurons[parentA.rowIndex][i][j];
                    Neuron neuronB = oldNeurons[parentB.rowIndex][i][j];
                    double newThreshold = Form1.R.Next(100) < 10 ? Form1.R.Next(-1 * Neuron.divisor, Neuron.divisor) / (double)Neuron.divisor : (Form1.R.Next(2) == 0 ? neuronA.threshold : neuronB.threshold);
                    double[] neuronWeights = new double[(i == 0 ? 784 : oldNeurons[parentA.rowIndex][i - 1].Length)];
                    for (int k = 0; k < prevLayerSize; k++)
                    {
                        neuronWeights[k] = Form1.R.Next(100) < 10 ? Form1.R.Next(-1 * Neuron.divisor, Neuron.divisor) / (double)Neuron.divisor : (Form1.R.Next(2) == 0 ? oldWeights[parentA.rowIndex][i][j][k] : oldWeights[parentB.rowIndex][i][j][k]);
                    }
                    newNeurons[rowIndex][i][j] = new Neuron(newThreshold, j, newWeights[rowIndex][i], neuronWeights);
                }
                prevLayerSize = layerSize;
            }
        }
        public NeuralNetwork(NeuralNetwork original, int rowIndex, double[][][][] newWeights, Neuron[][][] newNeurons, double[][][] newInputs, double[][][][] oldWeights, Neuron[][][] oldNeurons, double[][][] oldInputs)
        {
            this.rowIndex = rowIndex;
//            guesses = original.guesses;
//            actual = original.actual;
            //mutationRandom = new Random();
            newInputs[rowIndex] = oldInputs[original.rowIndex];
            newNeurons[rowIndex] = oldNeurons[original.rowIndex];
            newWeights[rowIndex] = oldWeights[original.rowIndex];
            testsDone = original.testsDone;
            correct = original.correct;
            sumCost = original.sumCost;
        }
        public double[] GetOutputs(double[] input, double[][][][] allWeights, Neuron[][][] allNeurons, double[][][] allInputs)
        {
            allInputs[rowIndex][0] = input;
            for (int i = 0; i < allNeurons[rowIndex].Length; i++)
            {
                for (int j = 0; j < allNeurons[rowIndex][i].Length; j++)
                {
                    allInputs[rowIndex][i + 1][j] = allNeurons[rowIndex][i][j].GetOutput(allInputs[rowIndex][i], allWeights[rowIndex][i]);
                }
            }
            return allInputs[rowIndex][allInputs[rowIndex].Length - 1];
        }
        public int MostConfidentOutput(double[] outputs)
        {
            return Array.IndexOf(outputs, outputs.Max());
        }
        public double GetCost(int desired, double[] output)
        {
            double totalCost = 0;
            for (int i = 0; i < output.Length; i++)
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
        public void Draw(Graphics g, Rectangle bounds, double[][][][] allWeights, Neuron[][][] allNeurons)
        {
            if (allNeurons[rowIndex] == null)
                return;
            Size padding = new Size(bounds.Width / (2 * allNeurons[rowIndex].Length), bounds.Height / (2 * allNeurons[rowIndex][0].Length));
            for (int i = 0; i < allNeurons[rowIndex].Length; i++)
            {
                for (int j = 0; j < allNeurons[rowIndex][i].Length; j++)
                {
                    double nodeSpacing = (bounds.Height - 2.0 * padding.Height) / (allNeurons[rowIndex][0].Length - 1);
                    int greyColor;
                    if (i > 0)
                    {
                        for (int k = 0; k < allNeurons[rowIndex][i - 1].Length; k++)
                        {
                            greyColor = (int)(allWeights[rowIndex][i][j][k] * 128 + 128);
                            g.DrawLine(new Pen(Color.FromArgb(greyColor, greyColor, greyColor)),
                                (int)(padding.Width * (1 + 2.0 * i)),
                                (int)(padding.Height + (bounds.Height - nodeSpacing * allNeurons[rowIndex][i].Length) / 2 + nodeSpacing * j),
                                (int)(padding.Width * (2.0 * i - 1)),
                                (int)(padding.Height + (bounds.Height - nodeSpacing * allNeurons[rowIndex][i - 1].Length) / 2 + nodeSpacing * k));
                        }
                    }
                    else if (nodeSpacing >= imageHeight) ;
                    {
                        Bitmap b = new Bitmap(imageWidth, imageHeight);
                        for (int k = 0; k < 784; k++)
                        {
                            greyColor = (int)((allWeights[rowIndex][0][j][k] - allNeurons[rowIndex][0][j].threshold) * 128 + 128);
                            greyColor = greyColor < 0 ? 0 : (greyColor > 255 ? 255 : greyColor);
                            b.SetPixel(k % imageWidth, k / imageWidth, Color.FromArgb(greyColor, greyColor, greyColor));
                        }
                        g.DrawImage(b, padding.Width - 40, (int)(padding.Height + (bounds.Height - nodeSpacing * allNeurons[rowIndex][0].Length) / 2 + nodeSpacing * j) - 14);
                    }
                    greyColor = (int)(allNeurons[rowIndex][i][j].threshold * 128 + 128);
                    g.FillEllipse(new SolidBrush(Color.FromArgb(greyColor, greyColor, greyColor)),
                        (int)(padding.Width * (1 + 2.0 * i)) - 5, (int)(padding.Height + (bounds.Height - nodeSpacing * allNeurons[rowIndex][i].Length) / 2 + nodeSpacing * j) - 5,
                        10, 10);
                }
            }
            if (testsDone > 0)
            {
                g.DrawString("" + (int)(correct / (double)testsDone * 100.0) + "% correct", new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 10);
            }
            g.DrawString("Generation " + Form1.generation, new Font(FontFamily.GenericMonospace, 20), Brushes.Black, 10, 40);
            Form1.generation++;
        }
    }
    public struct Neuron
    {
        public const int sectionLength = 4;
        public static readonly int divisor = 10000;
        public double threshold;
        double total;
        int rowIndex;
        public Neuron(double threshold, int rowIndex, double[][] allWeights, double[] weights) : this()
        {
            this.threshold = threshold;
            this.rowIndex = rowIndex;
            allWeights[rowIndex] = weights.Select(n => n).ToArray();
            total = 0;
        }
        public double GetOutput(double[] input, double[][] allWeights)
        {
            total = 0;
            for (int i = 0; i < input.Length; i++)
            {
                total += input[i] * allWeights[rowIndex][i];
            }
            total /= input.Length;
            return total - threshold > 0 ? (total - threshold < 1 ? total - threshold : 1) : 0;
        }
    }
    public struct Digit
    {
        public int value;
        public double[] pixelValues;
        public Digit(string path)
        {
            value = int.Parse(Directory.GetParent(path).Name);
            Bitmap image = new Bitmap(path);
            pixelValues = new double[image.Width * image.Height];
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