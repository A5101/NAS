using Microsoft.Win32;
using NAS.Core.NeuralNetworks;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Path = System.IO;

namespace NASTest;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>

public partial class MainWindow : Window
{
    private bool isDrawing = false;
    private Point lastPoint;
    private Brush currentBrush = Brushes.Black;
    private double brushSize = 8;
    private DynamicCNN loadedModel;
    private string[] classLabels = "А,Б,В,Г,Д,Е,Ё,Ж,З,И,Й,К,Л,М,Н,О,П,Р,С,Т,У,Ф,Х,Ц,Ч,Ш,Щ,Ъ,Ы,Ь,Э,Ю,Я".Split(',');

    public MainWindow()
    {
        InitializeComponent();
        UpdateModelStatus();
    }

    #region Рисование
    private void Canvas_MouseDown(object sender, MouseButtonEventArgs e)
    {
        isDrawing = true;
        lastPoint = e.GetPosition(drawingCanvas);
    }

    private void Canvas_MouseMove(object sender, MouseEventArgs e)
    {
        if (!isDrawing) return;

        Point currentPoint = e.GetPosition(drawingCanvas);

        // Рисуем линию между точками
        Line line = new Line
        {
            X1 = lastPoint.X,
            Y1 = lastPoint.Y,
            X2 = currentPoint.X,
            Y2 = currentPoint.Y,
            Stroke = currentBrush,
            StrokeThickness = brushSize,
            StrokeStartLineCap = PenLineCap.Round,
            StrokeEndLineCap = PenLineCap.Round
        };

        drawingCanvas.Children.Add(line);
        lastPoint = currentPoint;
    }

    private void Canvas_MouseUp(object sender, MouseButtonEventArgs e)
    {
        isDrawing = false;
    }

    private void Canvas_MouseLeave(object sender, MouseEventArgs e)
    {
        isDrawing = false;
    }

    private void BtnClear_Click(object sender, RoutedEventArgs e)
    {
        drawingCanvas.Children.Clear();
        tbResult.Text = "Нарисуйте букву и нажмите 'Распознать'";
        lbProbabilities.ItemsSource = null;
    }

    private void BtnLoadModel_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var openDialog = new OpenFileDialog
            {
                Filter = "PyTorch model files (*.pth)|*.pth",
                Title = "Выберите файл весов модели"
            };

            if (openDialog.ShowDialog() == true)
            {
                string weightsPath = openDialog.FileName;
                string architecturePath = System.IO.Path.Combine(
                    Path.GetDirectoryName(weightsPath),
                    Path.GetFileNameWithoutExtension(weightsPath) + "_architecture.json"
                );

                if (!File.Exists(architecturePath))
                {
                    // Пробуем найти файл архитектуры вручную
                    var archDialog = new OpenFileDialog
                    {
                        Filter = "JSON files (*.json)|*.json",
                        Title = "Выберите файл архитектуры модели",
                        InitialDirectory = Path.GetDirectoryName(weightsPath)
                    };

                    if (archDialog.ShowDialog() == true)
                    {
                        architecturePath = archDialog.FileName;
                    }
                    else
                    {
                        MessageBox.Show("Файл архитектуры не найден!", "Ошибка",
                            MessageBoxButton.OK, MessageBoxImage.Error);
                        return;
                    }
                }

                loadedModel = DynamicCNN.Load(weightsPath, architecturePath);
                UpdateModelStatus();

                MessageBox.Show($"Модель успешно загружена!\nКлассов: {classLabels.Length}",
                    "Успех", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Ошибка загрузки модели: {ex.Message}", "Ошибка",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void UpdateModelStatus()
    {
        if (loadedModel != null)
        {
            tbModelStatus.Text = "Модель загружена ✓";
            tbModelStatus.Foreground = Brushes.Green;
            btnRecognize.IsEnabled = true;
        }
        else
        {
            tbModelStatus.Text = "Модель не загружена";
            tbModelStatus.Foreground = Brushes.Red;
            btnRecognize.IsEnabled = false;
        }
    }
    #endregion

    #region Распознавание
    private void BtnRecognize_Click(object sender, RoutedEventArgs e)
    {
        if (loadedModel == null)
        {
            MessageBox.Show("Сначала загрузите модель!", "Ошибка",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (drawingCanvas.Children.Count == 0)
        {
            MessageBox.Show("Нарисуйте букву для распознавания!", "Ошибка",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        try
        {
            // Конвертируем рисунок в тензор
            Tensor inputTensor = ConvertCanvasToTensor();

            // Распознаем
            Tensor output = loadedModel.forward(inputTensor);
            var probabilities = output.softmax(1);

            // Получаем результаты
            ProcessRecognitionResults(probabilities);

            // Очищаем тензоры
            inputTensor.Dispose();
            output.Dispose();
            probabilities.Dispose();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Ошибка распознавания: {ex.Message}", "Ошибка",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private Tensor ConvertCanvasToTensor()
    {
        // Создаем RenderTargetBitmap из Canvas
        RenderTargetBitmap rtb = new RenderTargetBitmap(
            (int)drawingCanvas.ActualWidth,
            (int)drawingCanvas.ActualHeight,
            96, 96, PixelFormats.Pbgra32);

        rtb.Render(drawingCanvas);

        // Конвертируем в массив пикселей
        int stride = rtb.PixelWidth * 4;
        byte[] pixels = new byte[rtb.PixelHeight * stride];
        rtb.CopyPixels(pixels, stride, 0);

        // Преобразуем в grayscale и нормализуем
        int width = rtb.PixelWidth;
        int height = rtb.PixelHeight;
        float[] data = new float[width * height];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * stride + x * 4;
                byte b = pixels[index];
                byte g = pixels[index + 1];
                byte r = pixels[index + 2];

                // Конвертируем в grayscale и инвертируем (белый фон -> черный)
                float gray = 1.0f - (r * 0.3f + g * 0.59f + b * 0.11f) / 255.0f;
                data[y * width + x] = gray;
            }
        }

        // Создаем тензор [1, 1, height, width]
        return torch.tensor(data, torch.float32)
                    .reshape(new long[] { 1, 1, height, width });
    }

    private void ProcessRecognitionResults(Tensor probabilities)
    {
        var probArray = probabilities.data<float>().ToArray();

        // Находим лучший результат
        int bestClass = 0;
        float bestProb = 0;

        for (int i = 0; i < probArray.Length; i++)
        {
            if (probArray[i] > bestProb)
            {
                bestProb = probArray[i];
                bestClass = i;
            }
        }

        // Обновляем интерфейс
        string recognizedLetter = bestClass < classLabels.Length ?
            classLabels[bestClass] : $"Class {bestClass}";

        tbResult.Text = $"Распознано: {recognizedLetter}\n" +
                       $"Вероятность: {bestProb * 100:F1}%";

        // Показываем все вероятности
        var probList = new List<ProbabilityItem>();
        for (int i = 0; i < Math.Min(probArray.Length, classLabels.Length); i++)
        {
            probList.Add(new ProbabilityItem
            {
                Letter = classLabels[i],
                Probability = probArray[i] * 100,
                Percentage = $"{probArray[i] * 100:F1}%"
            });
        }

        // Сортируем по убыванию вероятности
        lbProbabilities.ItemsSource = probList
            .OrderByDescending(p => p.Probability)
            .Take(10) // Топ-10 результатов
            .ToList();
    }
    #endregion

    #region Настройки рисования
    private void ColorComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (colorComboBox.SelectedItem is ComboBoxItem item)
        {
            currentBrush = item.Foreground.Clone();
        }
    }

    private void BrushSizeSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        brushSize = brushSizeSlider.Value;
    }
    #endregion
}

// Класс для отображения вероятностей
public class ProbabilityItem
{
    public string Letter { get; set; } = "";
    public double Probability { get; set; }
    public string Percentage { get; set; } = "";
}
}