using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace textToPictures
{
    public partial class Form1 : Form
    {
        public static string neuralNetPath = Path.Combine(Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, @"..\..\..\..")), @"Neural Network\");
        public string textPath = Path.Combine(neuralNetPath, @"Numbers\numbers");
        public string text;
        public string outPath = Path.Combine(neuralNetPath, @"Images\");
        public Form1()
        {
            InitializeComponent();
            for (int i = 1; i <= 6; i++)
            {
                text = File.ReadAllText(textPath + i + ".txt");
                int count = 10000 * (i - 1);
                while (text != string.Empty)
                {
                    string number = text.Substring(0, 3);
                    text = text.Substring(3);
                    textToBitmap(text.Substring(0, 2352)).Save(outPath + number + "\\" + count + ".bmp", ImageFormat.Bmp);
                    text = text.Substring(2354);
                    count++;
                }
            }
        }
        public Bitmap textToBitmap(string text)
        {
            Bitmap b = new Bitmap(28, 28);
            int place = 0;
            while(text != string.Empty)
            {
                int pixelValue = int.Parse(text.Substring(0, 3));
                b.SetPixel(place % 28, place / 28, Color.FromArgb(pixelValue, pixelValue, pixelValue));
                place++;
                text = text.Substring(3);
            }
            return b;
        }
    }
}
