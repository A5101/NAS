using NAS.Core.NeuralNetworks;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using static TorchSharp.torch;
using TorchSharp;

namespace NASDemo
{
    /// <summary>
    /// Логика взаимодействия для DrawWindow.xaml
    /// </summary>
    public partial class DrawWindow : Window
    {
        private Point _lastPoint;
        private bool _isDrawing = false;
        private bool _isErasing = false;

        public DrawWindow()
        {
            InitializeComponent();

            // Инициализация Canvas
            drawCanvas.Width = 400;
            drawCanvas.Height = 400;
        }

        // Свойство для получения изображения с Canvas
        public System.Windows.Media.Imaging.BitmapSource DrawnImage
        {
            get
            {
                if (drawCanvas.Children.Count == 0)
                    return null;

                // Создаем копию BitmapSource в том же потоке
                RenderTargetBitmap rtb = new RenderTargetBitmap(
                    (int)drawCanvas.Width, (int)drawCanvas.Height,
                    96, 96, PixelFormats.Pbgra32);

                rtb.Render(drawCanvas);

                // Создаем замороженную копию, которую можно использовать в других потоках
                var frozenBitmap = new System.Windows.Media.Imaging.BitmapImage();
                var encoder = new System.Windows.Media.Imaging.PngBitmapEncoder();
                encoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(rtb));

                using (var stream = new System.IO.MemoryStream())
                {
                    encoder.Save(stream);
                    stream.Seek(0, System.IO.SeekOrigin.Begin);

                    frozenBitmap.BeginInit();
                    frozenBitmap.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
                    frozenBitmap.StreamSource = stream;
                    frozenBitmap.EndInit();
                    frozenBitmap.Freeze(); // Важно: замораживаем для использования в других потоках
                }

                return frozenBitmap;
            }
        }

        public byte[] GetDrawnImageBytes()
        {
            if (drawCanvas.Children.Count == 0)
                return null;

            RenderTargetBitmap rtb = new RenderTargetBitmap(
                (int)drawCanvas.Width, (int)drawCanvas.Height,
                96, 96, PixelFormats.Pbgra32);

            rtb.Render(drawCanvas);

            var encoder = new System.Windows.Media.Imaging.PngBitmapEncoder();
            encoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(rtb));

            using (var stream = new System.IO.MemoryStream())
            {
                encoder.Save(stream);
                return stream.ToArray();
            }
        }

        public bool HasDrawing => drawCanvas.Children.Count > 0;

        private void DrawCanvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                _isDrawing = true;
                _isErasing = false;
                _lastPoint = e.GetPosition(drawCanvas);
            }
            else if (e.RightButton == MouseButtonState.Pressed)
            {
                _isErasing = true;
                _isDrawing = false;
                _lastPoint = e.GetPosition(drawCanvas);
            }
        }

        private void DrawCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if ((_isDrawing || _isErasing) && e.LeftButton == MouseButtonState.Pressed)
            {
                Point currentPoint = e.GetPosition(drawCanvas);

                var line = new Line
                {
                    Stroke = _isDrawing ? Brushes.Black : Brushes.White,
                    StrokeThickness = _isDrawing ? 8 : 12,
                    StrokeStartLineCap = PenLineCap.Round,
                    StrokeEndLineCap = PenLineCap.Round,
                    X1 = _lastPoint.X,
                    Y1 = _lastPoint.Y,
                    X2 = currentPoint.X,
                    Y2 = currentPoint.Y
                };

                drawCanvas.Children.Add(line);
                _lastPoint = currentPoint;
            }
        }

        private void DrawCanvas_MouseUp(object sender, MouseButtonEventArgs e)
        {
            _isDrawing = false;
            _isErasing = false;
        }

        private void btnClear_Click(object sender, RoutedEventArgs e)
        {
            drawCanvas.Children.Clear();
        }

        private void btnOk_Click(object sender, RoutedEventArgs e)
        {
            if (drawCanvas.Children.Count == 0)
            {
                MessageBox.Show("Нарисуйте букву перед сохранением", "Информация",
                    MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            this.DialogResult = true;
            this.Close();
        }

        private void btnCancel_Click(object sender, RoutedEventArgs e)
        {
            this.DialogResult = false;
            this.Close();
        }
    }
}
