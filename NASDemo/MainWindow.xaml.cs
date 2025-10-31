using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
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
using static NAS.NAS.Controllers.RandomNASController;
using NAS.Core.Architecture;
using NAS.Data;
using NAS.NAS.Controllers;
using NAS.NAS.Models;
using static TorchSharp.torch;
using NAS.Core.Training;
using NAS.Core.NeuralNetworks;
using Microsoft.Win32;
using System.IO;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TorchSharp;

namespace NASDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private GeneticNASController _geneticController;
        private RandomNASController _randomController;
        private CyrillicDataLoader _dataLoader;
        private List<NASResult> _architecturesHistory;
        private bool _isSearchRunning = false;
        private CancellationTokenSource _cancellationTokenSource;

        public ObservableCollection<ArchitectureViewModel> Architectures { get; set; }
        public ObservableCollection<LayerInfo> CurrentLayers { get; set; }

        public MainWindow()
        {
            InitializeComponent();
            InitializeData();
            UpdateGeneticSettingsVisibility();
        }

        private void InitializeData()
        {
            Architectures = new ObservableCollection<ArchitectureViewModel>();
            CurrentLayers = new ObservableCollection<LayerInfo>();
            _architecturesHistory = new List<NASResult>();

            lbArchitectures.ItemsSource = Architectures;
            lbLayers.ItemsSource = CurrentLayers;

            // Инициализация контроллеров
            var device = cuda.is_available() ? CUDA : CPU;
            _geneticController = new GeneticNASController(imageSize: 64, device: device);
            _randomController = new RandomNASController(imageSize: 64, device: device);

            InitializeDataLoader();
        }

        private async void InitializeDataLoader()
        {
            try
            {
                tbProgressStatus.Text = "Загрузка данных...";
                await Task.Run(() =>
                {
                    _dataLoader = new CyrillicDataLoader("Cyrillic", imageSize: 64);
                });
                tbProgressStatus.Text = "Данные загружены";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка загрузки данных: {ex.Message}", "Ошибка",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                tbProgressStatus.Text = "Ошибка загрузки данных";
            }
        }

        private void UpdateGeneticSettingsVisibility()
        {
            if (grpGeneticSettings == null) return;

            var isGenetic = cmbSearchAlgorithm.SelectedIndex == 1;
            grpGeneticSettings.Visibility = isGenetic ? Visibility.Visible : Visibility.Collapsed;
        }

        private void cmbSearchAlgorithm_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            UpdateGeneticSettingsVisibility();
        }

        private async void btnStartSearch_Click(object sender, RoutedEventArgs e)
        {
            if (_dataLoader == null)
            {
                MessageBox.Show("Сначала загрузите данные", "Ошибка",
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            if (!ValidateInputs()) return;

            _isSearchRunning = true;
            _cancellationTokenSource = new CancellationTokenSource();
            UpdateControlState();

            try
            {
                await RunNASearch(_cancellationTokenSource.Token);
            }
            catch (OperationCanceledException)
            {
                tbProgressStatus.Text = "Поиск отменен";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка выполнения поиска: {ex.Message}", "Ошибка",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                _isSearchRunning = false;
                UpdateControlState();
            }
        }

        private bool ValidateInputs()
        {
            if (!int.TryParse(txtMinLayers.Text, out int minLayers) || minLayers < 3)
            {
                MessageBox.Show("Некорректное минимальное количество слоев", "Ошибка");
                return false;
            }

            if (!int.TryParse(txtMaxLayers.Text, out int maxLayers) || maxLayers < minLayers)
            {
                MessageBox.Show("Некорректное максимальное количество слоев", "Ошибка");
                return false;
            }

            return true;
        }

        private async Task RunNASearch(CancellationToken cancellationToken)
        {
            var config = GetSearchConfig();
            NASResult bestResult = null;

            if (cmbSearchAlgorithm.SelectedIndex == 0) // Случайный поиск
            {
                await Dispatcher.InvokeAsync(() =>
                {
                    tbProgressStatus.Text = "Запуск случайного поиска...";
                });

                var progress = new Progress<RandomNASController.ArchitectureResult>(individual =>
                {
                    var result = new NASResult(individual, "Random");
                    AddArchitectureToHistory(result);
                });
                var randomResult = await Task.Run(() =>
                    _randomController.Search(
                        dataLoader: _dataLoader,
                        batchSize: config.BatchSize,
                        imageSize: config.ImageSize,
                        numTrials: config.NumTrials,
                        minLayers: config.MinLayers,
                        maxLayers: config.MaxLayers,
                        epochsPerTrial: config.EpochsPerTrial,
                        progress: progress
                    ), cancellationToken);

                if (randomResult != null)
                {
                    bestResult = new NASResult(randomResult, "Random");
                }
            }
            else // Генетический алгоритм
            {
                var geneticConfig = new GeneticConfig
                {
                    PopulationSize = config.PopulationSize,
                    Generations = config.Generations,
                    CrossoverRate = config.CrossoverRate,
                    MutationRate = config.MutationRate,
                    EliteRatio = 0.1,
                    MinLayers = config.MinLayers,
                    MaxLayers = config.MaxLayers,
                    EpochsPerEvaluation = config.EpochsPerTrial
                };

                _geneticController = new GeneticNASController(imageSize: 64, config: geneticConfig);

                await Dispatcher.InvokeAsync(() =>
                {
                    tbProgressStatus.Text = "Запуск генетического алгоритма...";
                });

                // Подписка на события прогресса для ГА
                var progress = new Progress<GeneticNASController.ArchitectureIndividual>(individual =>
                {
                    var result = new NASResult(individual, "Genetic");
                    AddArchitectureToHistory(result);

                    Dispatcher.Invoke(() =>
                    {
                        tbProgressStatus.Text = $"Генетический алгоритм: Поколение {individual.Generation}, Лучшая точность: {individual.Accuracy:F2}%";
                        pbSearchProgress.Value = (individual.Generation * 100.0) / config.Generations;
                    });
                });

                var geneticResult = await Task.Run(() =>
                    _geneticController.Evolve(
                        _dataLoader,
                        config.BatchSize,
                        config.ImageSize,
                        progress
                    ), cancellationToken);

                if (geneticResult != null)
                {
                    bestResult = new NASResult(geneticResult, "Genetic");
                }
            }

            if (bestResult != null)
            {
                //AddArchitectureToHistory(bestResult);
                //DisplayArchitecture(bestResult);

                await Dispatcher.InvokeAsync(() =>
                {
                    tbProgressStatus.Text = $"Поиск завершен. Лучшая точность: {bestResult.Accuracy:F2}%";
                    pbSearchProgress.Value = 100;
                });
            }
        }

        private SearchConfig GetSearchConfig()
        {
            return new SearchConfig
            {
                NumTrials = int.TryParse(txtTrialsCount.Text, out int trials) ? trials : 50,
                EpochsPerTrial = int.TryParse(txtEpochsPerTrial.Text, out int epochs) ? epochs : 10,
                MinLayers = int.TryParse(txtMinLayers.Text, out int minLayers) ? minLayers : 4,
                MaxLayers = int.TryParse(txtMaxLayers.Text, out int maxLayers) ? maxLayers : 15,
                PopulationSize = int.TryParse(txtPopulationSize.Text, out int popSize) ? popSize : 20,
                Generations = int.TryParse(txtGenerations.Text, out int generations) ? generations : 30,
                MutationRate = sldMutationRate?.Value ?? 0.3,
                CrossoverRate = sldCrossoverRate?.Value ?? 0.7,
                BatchSize = 32,
                ImageSize = 64
            };
        }

        private void AddArchitectureToHistory(NASResult result)
        {
            var viewModel = new ArchitectureViewModel(result);

            Dispatcher.Invoke(() =>
            {
                Architectures.Add(viewModel);
                _architecturesHistory.Add(result);

                // Автоматическая сортировка по точности
                var sorted = Architectures.OrderByDescending(a => a.Accuracy).ToList();
                Architectures.Clear();
                foreach (var item in sorted)
                {
                    Architectures.Add(item);
                }

                // Автоматический выбор лучшей архитектуры
                if (lbArchitectures.SelectedIndex == -1 && Architectures.Count > 0)
                {
                    lbArchitectures.SelectedIndex = 0;
                }
            });
        }

        private void DisplayArchitecture(NASResult result)
        {
            // Обновление деталей архитектуры
            tbArchitectureName.Text = result.Algorithm == "Genetic"
                ? $"🧬 {result.Architecture.Name} (Поколение {result.Generation})"
                : $"🎲 {result.Architecture.Name}";

            tbTotalLayers.Text = $"Слоев: {result.Architecture.Layers.Count}";
            tbTotalParameters.Text = $"Параметров: {result.Parameters:N0}";
            tbTrainingTime.Text = $"Время обучения: {result.TrainingTime:F1}с";
            tbAccuracy.Text = $"Точность: {result.Accuracy:F2}%";
            tbInferenceTime.Text = $"Алгоритм: {result.Algorithm}";
            tbFitness.Text = $"Фитнес: {result.Accuracy / 100.0:F3}";

            // Обновление статистики поиска
            tbSearchTime.Text = $"Время поиска: {result.Timestamp:HH:mm:ss}";
            tbArchitecturesEvaluated.Text = $"Оценено архитектур: {_architecturesHistory.Count}";
            tbBestGeneration.Text = result.Algorithm == "Genetic"
                ? $"Лучшее поколение: {result.Generation}"
                : "Метод: Случайный поиск";

            // Обновление списка слоев
            CurrentLayers.Clear();
            foreach (var layer in result.Architecture.Layers)
            {
                CurrentLayers.Add(new LayerInfo(layer));
            }

            // Визуализация архитектуры
            DrawArchitectureOnCanvas(result.Architecture);

            // Отрисовка графиков обучения
            DrawTrainingGraphs(result);
        }

        private void DrawArchitectureOnCanvas(ConcreteArchitecture architecture)
        {
            cnvArchitecture.Children.Clear();

            double x = 50;
            double y = 50;
            double layerWidth = 120;
            double layerHeight = 60;
            double horizontalSpacing = 10; // Расстояние между слоями по горизонтали

            for (int i = 0; i < architecture.Layers.Count; i++)
            {
                var layer = architecture.Layers[i];

                // Рисуем прямоугольник слоя
                var rect = new Rectangle
                {
                    Width = layerWidth,
                    Height = layerHeight,
                    Fill = GetLayerColor(layer.Type),
                    Stroke = Brushes.Black,
                    StrokeThickness = 1
                };

                Canvas.SetLeft(rect, x);
                Canvas.SetTop(rect, y);
                cnvArchitecture.Children.Add(rect);

                // Добавляем текст с описанием слоя
                var textBlock = new TextBlock
                {
                    Text = GetLayerShortDescription(layer),
                    FontSize = 9,
                    TextWrapping = TextWrapping.Wrap,
                    Width = layerWidth - 10,
                    TextAlignment = TextAlignment.Center,
                    Foreground = Brushes.Black
                };

                Canvas.SetLeft(textBlock, x + 5);
                Canvas.SetTop(textBlock, y + 5);
                cnvArchitecture.Children.Add(textBlock);

                // Добавляем номер слоя сверху
                var layerNumber = new TextBlock
                {
                    Text = $"{i + 1}",
                    FontSize = 8,
                    FontWeight = FontWeights.Bold,
                    Foreground = Brushes.DarkBlue,
                    Background = Brushes.LightYellow,
                    Padding = new Thickness(2)
                };

                Canvas.SetLeft(layerNumber, x + layerWidth / 2 - 5);
                Canvas.SetTop(layerNumber, y - 15);
                cnvArchitecture.Children.Add(layerNumber);

                x += layerWidth + horizontalSpacing;

                // Если достигли правого края, переходим на следующую строку
                if (x + layerWidth > cnvArchitecture.ActualWidth - 50)
                {
                    x = 50;
                    y += layerHeight + 80; // Увеличиваем вертикальное расстояние между строками

                    // Добавляем разделитель между строками (опционально)
                    if (y > 50)
                    {
                        var separator = new Line
                        {
                            X1 = 30,
                            Y1 = y - 40,
                            X2 = cnvArchitecture.ActualWidth - 30,
                            Y2 = y - 40,
                            Stroke = Brushes.LightGray,
                            StrokeThickness = 1,
                            StrokeDashArray = new DoubleCollection { 5, 5 }
                        };
                        cnvArchitecture.Children.Add(separator);
                    }
                }
            }

            // Добавляем подписи Input и Output
            DrawInputOutputLabels(cnvArchitecture, architecture);
        }

        // Метод для рисования стрелки
        private void DrawArrow(Canvas canvas, double x, double y)
        {
            var arrow = new Polygon
            {
                Points = new PointCollection
        {
            new Point(x, y),
            new Point(x - 8, y - 4),
            new Point(x - 8, y + 4)
        },
                Fill = Brushes.Gray,
                Stroke = Brushes.Gray
            };

            canvas.Children.Add(arrow);
        }

        // Метод для добавления подписей Input/Output
        private void DrawInputOutputLabels(Canvas canvas, ConcreteArchitecture architecture)
        {
            if (architecture.Layers.Count == 0) return;

            // Input label
            var inputLabel = new TextBlock
            {
                Text = "Input\n1×64×64",
                FontSize = 8,
                FontWeight = FontWeights.Bold,
                Foreground = Brushes.DarkGreen,
                TextAlignment = TextAlignment.Center,
                Background = Brushes.LightGreen,
                Padding = new Thickness(3)
            };

            Canvas.SetLeft(inputLabel, 10);
            Canvas.SetTop(inputLabel, 50);
            canvas.Children.Add(inputLabel);

            // Output label
            var outputLabel = new TextBlock
            {
                Text = $"Output\n{((architecture.Layers.Last() is OutputLayer output) ? output.NumClasses : "?")} classes",
                FontSize = 8,
                FontWeight = FontWeights.Bold,
                Foreground = Brushes.DarkRed,
                TextAlignment = TextAlignment.Center,
                Background = Brushes.LightCoral,
                Padding = new Thickness(3)
            };

            // Позиционируем output label рядом с последним слоем
            var lastLayer = canvas.Children
                .OfType<Rectangle>()
                .LastOrDefault();

            if (lastLayer != null)
            {
                double lastX = Canvas.GetLeft(lastLayer) + lastLayer.Width + 10;
                double lastY = Canvas.GetTop(lastLayer);

                Canvas.SetLeft(outputLabel, lastX);
                Canvas.SetTop(outputLabel, lastY);
                canvas.Children.Add(outputLabel);
            }
        }

        // Улучшенный метод для краткого описания слоев
        private string GetLayerShortDescription(Layer layer)
        {
            return layer switch
            {
                ConvLayer conv => $"Conv2D\n{conv.Filters} filters\n{conv.KernelSize}×{conv.KernelSize}\n{conv.Activation}",
                PoolingLayer pool => $"{pool.PoolType.ToUpper()} Pool\n{pool.PoolSize}×{pool.PoolSize}",
                FullyConnectedLayer fc => $"Dense\n{fc.Units} units\n{(fc.DropoutRate > 0 ? $"Dropout {fc.DropoutRate:P0}" : "")}",
                OutputLayer output => $"Output\n{output.NumClasses} classes\nSoftmax",
                CustomLayer custom when custom.Type == "flatten" => "Flatten",
                CustomLayer custom => $"{custom.Type}\n{custom.Name}",
                _ => layer.GetType().Name
            };
        }

        // Улучшенная цветовая схема
        private Brush GetLayerColor(string layerType)
        {
            return layerType switch
            {
                "conv" => new LinearGradientBrush(Colors.LightBlue, Colors.SteelBlue, 45),
                "pool" => new LinearGradientBrush(Colors.LightGreen, Colors.SeaGreen, 45),
                "dense" => new LinearGradientBrush(Colors.LightPink, Colors.DeepPink, 45),
                "output" => new LinearGradientBrush(Colors.Gold, Colors.Orange, 45),
                "flatten" => new LinearGradientBrush(Colors.Plum, Colors.MediumPurple, 45),
                _ => new LinearGradientBrush(Colors.LightGray, Colors.Gray, 45)
            };
        }

        private void DrawTrainingGraphs(NASResult result)
        {
            if (result?.TrainingHistory == null || !result.TrainingHistory.Any())
            {
                DrawNoDataMessage(cnvTrainingGraph, "Нет данных об обучении");
                DrawNoDataMessage(cnvMetricsGraph, "Нет данных об обучении");
                return;
            }

            // Очищаем канвасы
            cnvTrainingGraph.Children.Clear();
            cnvMetricsGraph.Children.Clear();

            // Рисуем графики точности и потерь
            DrawAccuracyAndLossGraph(result.TrainingHistory);

            // Рисуем графики метрик и анализа обучения
            DrawMetricsAndAnalysisGraph(result.TrainingHistory);
        }

        private void DrawAccuracyAndLossGraph(List<TrainingEpoch> history)
        {
            if (history == null || !history.Any()) return;

            const double margin = 40;
            double canvasWidth = cnvTrainingGraph.ActualWidth - 2 * margin;
            double canvasHeight = cnvTrainingGraph.ActualHeight - 2 * margin;

            // Находим минимальные и максимальные значения для нормализации
            double maxAccuracy = history.Max(e => Math.Max(e.TrainAccuracy, e.ValAccuracy));
            double minAccuracy = history.Min(e => Math.Min(e.TrainAccuracy, e.ValAccuracy));
            double maxLoss = history.Max(e => Math.Max(e.TrainLoss, e.ValLoss));
            double minLoss = history.Min(e => Math.Min(e.TrainLoss, e.ValLoss));

            // Добавляем немного места по краям
            maxAccuracy += 2;
            minAccuracy = Math.Max(0, minAccuracy - 2);
            maxLoss += maxLoss * 0.1;
            minLoss = Math.Max(0, minLoss - minLoss * 0.1);

            // Рисуем оси
            DrawAxes(cnvTrainingGraph, margin, canvasWidth, canvasHeight, "Эпохи", "Точность (%) / Потери");

            // Рисуем сетку
            DrawGrid(cnvTrainingGraph, margin, canvasWidth, canvasHeight, history.Count, 5);

            // Рисуем линии точности
            DrawLineGraph(cnvTrainingGraph, history, e => e.TrainAccuracy,
                         Brushes.DodgerBlue, "Train Accuracy", margin, canvasWidth, canvasHeight,
                         0, history.Count, minAccuracy, maxAccuracy);

            DrawLineGraph(cnvTrainingGraph, history, e => e.ValAccuracy,
                         Brushes.LimeGreen, "Val Accuracy", margin, canvasWidth, canvasHeight,
                         0, history.Count, minAccuracy, maxAccuracy);

            DrawLineGraph(cnvTrainingGraph, history, e => e.TrainLoss,
                         Brushes.OrangeRed, "Train Loss", margin, canvasWidth, canvasHeight,
                         0, history.Count, minLoss, maxLoss);

            DrawLineGraph(cnvTrainingGraph, history, e => e.ValLoss,
                         Brushes.Purple, "Val Loss", margin, canvasWidth, canvasHeight,
                         0, history.Count, minLoss, maxLoss);

            // Добавляем легенду
            DrawLegend(cnvTrainingGraph, new[]
            {
        ("Train Accuracy", Brushes.DodgerBlue),
        ("Val Accuracy", Brushes.LimeGreen),
        ("Train Loss", Brushes.OrangeRed),
        ("Val Loss", Brushes.Purple)
    }, margin, canvasWidth);
        }

        private void DrawMetricsAndAnalysisGraph(List<TrainingEpoch> history)
        {
            if (history == null || !history.Any()) return;

            const double margin = 40;
            double canvasWidth = cnvMetricsGraph.ActualWidth - 2 * margin;
            double canvasHeight = cnvMetricsGraph.ActualHeight - 2 * margin;

            // Рассчитываем дополнительные метрики
            var metrics = history.Select(e => new
            {
                Epoch = e.Epoch,
                OverfittingIndicator = e.LossDifference, // Разница между train и val loss
                LearningSpeed = CalculateLearningSpeed(history, e.Epoch),
                GeneralizationGap = e.AccuracyDifference // Разница между val и train accuracy
            }).ToList();

            // Находим диапазоны для нормализации
            double maxOverfitting = metrics.Max(m => Math.Abs(m.OverfittingIndicator));
            double maxLearningSpeed = metrics.Max(m => Math.Abs(m.LearningSpeed));
            double maxGap = metrics.Max(m => Math.Abs(m.GeneralizationGap));

            // Рисуем оси
            DrawAxes(cnvMetricsGraph, margin, canvasWidth, canvasHeight, "Эпохи", "Метрики обучения");

            // Рисуем сетку
            DrawGrid(cnvMetricsGraph, margin, canvasWidth, canvasHeight, history.Count, 5);

            // Рисуем метрики
            DrawLineGraph(cnvMetricsGraph, metrics, m => m.OverfittingIndicator,
                         Brushes.Red, "Overfitting Indicator", margin, canvasWidth, canvasHeight,
                         0, metrics.Count, -maxOverfitting, maxOverfitting);

            DrawLineGraph(cnvMetricsGraph, metrics, m => m.LearningSpeed,
                         Brushes.Blue, "Learning Speed", margin, canvasWidth, canvasHeight,
                         0, metrics.Count, -maxLearningSpeed, maxLearningSpeed);

            DrawLineGraph(cnvMetricsGraph, metrics, m => m.GeneralizationGap,
                         Brushes.Green, "Generalization Gap", margin, canvasWidth, canvasHeight,
                         0, metrics.Count, -maxGap, maxGap);

            // Добавляем легенду
            DrawLegend(cnvMetricsGraph, new[] { ("Overfitting Indicator", Brushes.Red), ("Learning Speed", Brushes.Blue), ("Generalization Gap", Brushes.Green) }, margin, canvasWidth);

            // Добавляем анализ обучения
            // DrawTrainingAnalysis(cnvMetricsGraph, history, margin, canvasWidth, canvasHeight);
        }

        private double CalculateLearningSpeed(List<TrainingEpoch> history, int epoch, int window = 5)
        {
            if (epoch <= window) return 0;

            var recentEpochs = history.Where(e => e.Epoch > epoch - window && e.Epoch <= epoch).ToList();
            if (recentEpochs.Count < 2) return 0;

            var first = recentEpochs.First();
            var last = recentEpochs.Last();

            return (last.ValAccuracy - first.ValAccuracy) / (last.Epoch - first.Epoch);
        }

        // Вспомогательные методы для рисования графиков
        private void DrawAxes(Canvas canvas, double margin, double width, double height, string xLabel, string yLabel)
        {
            // Ось X
            var xAxis = new Line
            {
                X1 = margin,
                Y1 = margin + height,
                X2 = margin + width,
                Y2 = margin + height,
                Stroke = Brushes.Black,
                StrokeThickness = 2
            };
            canvas.Children.Add(xAxis);

            // Ось Y
            var yAxis = new Line
            {
                X1 = margin,
                Y1 = margin,
                X2 = margin,
                Y2 = margin + height,
                Stroke = Brushes.Black,
                StrokeThickness = 2
            };
            canvas.Children.Add(yAxis);

            // Подписи осей
            var xLabelText = new TextBlock
            {
                Text = xLabel,
                FontSize = 10,
                Foreground = Brushes.Black
            };
            Canvas.SetLeft(xLabelText, margin + width / 2 - 20);
            Canvas.SetTop(xLabelText, margin + height + 20);
            canvas.Children.Add(xLabelText);

            var yLabelText = new TextBlock
            {
                Text = yLabel,
                FontSize = 10,
                Foreground = Brushes.Black,
                RenderTransform = new RotateTransform(-90)
            };
            Canvas.SetLeft(yLabelText, 5);
            Canvas.SetTop(yLabelText, margin + height / 2 - 20);
            canvas.Children.Add(yLabelText);
        }

        private void DrawGrid(Canvas canvas, double margin, double width, double height, int xSteps, int ySteps)
        {
            // Вертикальные линии (эпохи)
            for (int i = 0; i <= xSteps; i++)
            {
                var line = new Line
                {
                    X1 = margin + (width * i / xSteps),
                    Y1 = margin,
                    X2 = margin + (width * i / xSteps),
                    Y2 = margin + height,
                    Stroke = Brushes.LightGray,
                    StrokeThickness = 0.5
                };
                canvas.Children.Add(line);

                var label = new TextBlock
                {
                    Text = (i * (xSteps > 0 ? xSteps / xSteps : 1)).ToString(),
                    FontSize = 8,
                    Foreground = Brushes.Gray
                };
                Canvas.SetLeft(label, margin + (width * i / xSteps) - 5);
                Canvas.SetTop(label, margin + height + 5);
                canvas.Children.Add(label);
            }

            // Горизонтальные линии (значения)
            for (int i = 0; i <= ySteps; i++)
            {
                var line = new Line
                {
                    X1 = margin,
                    Y1 = margin + (height * i / ySteps),
                    X2 = margin + width,
                    Y2 = margin + (height * i / ySteps),
                    Stroke = Brushes.LightGray,
                    StrokeThickness = 0.5
                };
                canvas.Children.Add(line);
            }
        }

        private void DrawLineGraph<T>(Canvas canvas, List<T> data, Func<T, double> valueSelector,
                                     Brush color, string label, double margin, double width, double height,
                                     double xMin, double xMax, double yMin, double yMax, bool isRightAxis = false)
        {
            if (data == null || !data.Any()) return;

            var points = new List<Point>();
            double xRange = xMax - xMin;
            double yRange = yMax - yMin;

            if (yRange == 0) yRange = 1; // Защита от деления на ноль

            for (int i = 0; i < data.Count; i++)
            {
                double x = margin + (width * (i - xMin) / xRange);
                double y = margin + height - (height * (valueSelector(data[i]) - yMin) / yRange);

                if (isRightAxis)
                {
                    x = margin + width - (width * (i - xMin) / xRange);
                }

                points.Add(new Point(x, y));
            }

            // Рисуем линию
            for (int i = 1; i < points.Count; i++)
            {
                var line = new Line
                {
                    X1 = points[i - 1].X,
                    Y1 = points[i - 1].Y,
                    X2 = points[i].X,
                    Y2 = points[i].Y,
                    Stroke = color,
                    StrokeThickness = 2
                };
                canvas.Children.Add(line);
            }

            // Рисуем точки на ключевых позициях
            for (int i = 0; i < points.Count; i += Math.Max(1, points.Count / 10))
            {
                var ellipse = new Ellipse
                {
                    Width = 4,
                    Height = 4,
                    Fill = color,
                    Stroke = Brushes.White,
                    StrokeThickness = 1
                };
                Canvas.SetLeft(ellipse, points[i].X - 2);
                Canvas.SetTop(ellipse, points[i].Y - 2);
                canvas.Children.Add(ellipse);
            }
        }

        private void DrawLegend(Canvas canvas, (string label, SolidColorBrush color)[] items, double margin, double width)
        {
            double top = margin;
            double left = margin + width - 100;

            foreach (var (label, color) in items)
            {
                var colorBox = new Rectangle
                {
                    Width = 12,
                    Height = 12,
                    Fill = color,
                    Stroke = Brushes.Black,
                    StrokeThickness = 1
                };
                Canvas.SetLeft(colorBox, left);
                Canvas.SetTop(colorBox, top);
                canvas.Children.Add(colorBox);

                var labelText = new TextBlock
                {
                    Text = label,
                    FontSize = 9,
                    Foreground = Brushes.Black
                };
                Canvas.SetLeft(labelText, left + 15);
                Canvas.SetTop(labelText, top - 2);
                canvas.Children.Add(labelText);

                top += 15;
            }
        }

        private void DrawTrainingAnalysis(Canvas canvas, List<TrainingEpoch> history, double margin, double width, double height)
        {
            if (history == null || history.Count < 10) return;

            var bestEpoch = history.OrderByDescending(e => e.ValAccuracy).First();
            var lastEpoch = history.Last();

            var analysisText = new TextBlock
            {
                Text = $"Анализ обучения:\n" +
                       $"Лучшая эпоха: {bestEpoch.Epoch}\n" +
                       $"Лучшая точность: {bestEpoch.ValAccuracy:F2}%\n" +
                       $"Финальная точность: {lastEpoch.ValAccuracy:F2}%\n" +
                       $"Разница: {bestEpoch.ValAccuracy - lastEpoch.ValAccuracy:+#.##;-#.##;0}%",
                FontSize = 9,
                Foreground = Brushes.DarkBlue,
                Background = Brushes.LightYellow,
                Padding = new Thickness(5)
            };

            Canvas.SetLeft(analysisText, margin + width - 150);
            Canvas.SetTop(analysisText, margin + 10);
            canvas.Children.Add(analysisText);
        }

        private void DrawNoDataMessage(Canvas canvas, string message)
        {
            var textBlock = new TextBlock
            {
                Text = message,
                FontSize = 14,
                Foreground = Brushes.Gray,
                HorizontalAlignment = HorizontalAlignment.Center,
                VerticalAlignment = VerticalAlignment.Center
            };

            var border = new Border
            {
                Child = textBlock,
                Background = Brushes.White,
                BorderBrush = Brushes.LightGray,
                BorderThickness = new Thickness(1),
                Width = canvas.ActualWidth,
                Height = canvas.ActualHeight
            };

            canvas.Children.Add(border);
        }

        private void lbArchitectures_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (lbArchitectures.SelectedItem is ArchitectureViewModel selected)
            {
                var result = _architecturesHistory.FirstOrDefault(a =>
                    a.Architecture.Name == selected.Name && a.Algorithm == selected.Algorithm);
                if (result != null)
                {
                    DisplayArchitecture(result);
                }
            }
        }

        private void btnStopSearch_Click(object sender, RoutedEventArgs e)
        {
            _cancellationTokenSource?.Cancel();
            _isSearchRunning = false;
            UpdateControlState();
            tbProgressStatus.Text = "Поиск остановлен";
        }

        private void UpdateControlState()
        {
            btnStartSearch.IsEnabled = !_isSearchRunning;
            btnStopSearch.IsEnabled = _isSearchRunning;
            pbSearchProgress.IsIndeterminate = _isSearchRunning;
        }

        private void btnSortByAccuracy_Click(object sender, RoutedEventArgs e)
        {
            var sorted = Architectures.OrderByDescending(a => a.Accuracy).ToList();
            Architectures.Clear();
            foreach (var item in sorted)
            {
                Architectures.Add(item);
            }
        }

        private async void btnDrawAndRecognize_Click(object sender, RoutedEventArgs e)
        {
            if (lbArchitectures.SelectedItem is ArchitectureViewModel selected)
            {
                var drawWindow = new DrawWindow();
                drawWindow.Owner = this;

                if (drawWindow.ShowDialog() == true && drawWindow.HasDrawing)
                {
                    try
                    {
                        btnDrawAndRecognize.IsEnabled = false;
                        btnDrawAndRecognize.Content = "⏳ Обработка...";

                        // Получаем изображение из окна рисования
                        var bitmapSource = drawWindow.DrawnImage;

                        if (bitmapSource != null)
                        {
                            // Конвертируем в тензор и распознаем
                            var results = await ProcessAndRecognizeImage(bitmapSource, selected.CNNModel);

                            // Отображаем результаты
                            DisplayRecognitionResults(results);
                        }
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"Ошибка распознавания: {ex.Message}", "Ошибка",
                            MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                    finally
                    {
                        btnDrawAndRecognize.IsEnabled = true;
                        btnDrawAndRecognize.Content = "✏️ Нарисовать и распознать";
                    }
                }
            }
        }
        private async Task<List<(string letter, double confidence)>> ProcessAndRecognizeImage(BitmapSource bitmapSource, DynamicCNN model)
        {
            return await Task.Run(() =>
            {
                // Конвертируем BitmapSource в тензор
                var inputTensor = ConvertBitmapToTensor(bitmapSource);

                model.eval();

                using (no_grad())
                {
                    var output = model.forward(inputTensor);
                    var probabilities = torch.nn.functional.softmax(output, dim: 1);
                    var probArray = probabilities.cpu().data<float>().ToArray();

                    var results = new List<(string, double confidence)>();

                    // Получаем метки классов
                    var classLabels = GetClassLabels();

                    for (int i = 0; i < probArray.Length && i < classLabels.Count; i++)
                    {
                        double confidence = probArray[i] * 100.0;
                        if (confidence > 0.1) // Показываем только вероятности > 0.1%
                        {
                            string letter = classLabels[i];
                            results.Add((letter, confidence));
                        }
                    }

                    output.Dispose();
                    probabilities.Dispose();
                    inputTensor.Dispose();

                    return results.OrderByDescending(r => r.confidence).Take(5).ToList();
                }
            });
        }

        private Tensor ConvertBitmapToTensor(BitmapSource bitmapSource)
        {
            // Способ 1: Через массив байтов (надежнее)
            byte[] bitmapData;

            if (bitmapSource is System.Windows.Media.Imaging.BitmapImage bitmapImage)
            {
                // Если это BitmapImage, получаем данные из StreamSource
                var encoder = new System.Windows.Media.Imaging.PngBitmapEncoder();
                encoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(bitmapSource));

                using (var stream = new System.IO.MemoryStream())
                {
                    encoder.Save(stream);
                    bitmapData = stream.ToArray();
                }
            }
            else
            {
                // Для других типов BitmapSource
                bitmapData = ConvertBitmapSourceToByteArray(bitmapSource);
            }

            return ProcessImageBytesToTensor(bitmapData);
        }

        private byte[] ConvertBitmapSourceToByteArray(BitmapSource bitmapSource)
        {
            // Создаем копию в текущем потоке
            var copiedBitmap = new System.Windows.Media.Imaging.RenderTargetBitmap(
                bitmapSource.PixelWidth, bitmapSource.PixelHeight,
                bitmapSource.DpiX, bitmapSource.DpiY,
                PixelFormats.Pbgra32);

            var drawingVisual = new DrawingVisual();
            using (var drawingContext = drawingVisual.RenderOpen())
            {
                drawingContext.DrawImage(bitmapSource, new Rect(0, 0, bitmapSource.PixelWidth, bitmapSource.PixelHeight));
            }

            copiedBitmap.Render(drawingVisual);

            // Кодируем в PNG
            var encoder = new System.Windows.Media.Imaging.PngBitmapEncoder();
            encoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(copiedBitmap));

            using (var stream = new System.IO.MemoryStream())
            {
                encoder.Save(stream);
                return stream.ToArray();
            }
        }

        private Tensor ProcessImageBytesToTensor(byte[] imageBytes)
        {
            using (var image = SixLabors.ImageSharp.Image.Load<Rgba32>(imageBytes))
            {
                // Преобразуем в grayscale и ресайзим до 64x64
                image.Mutate(x => x
                    .Resize(64, 64)
                    .Grayscale());

                // Создаем тензор [1, 1, 64, 64]
                var tensor = torch.zeros(new long[] { 1, 1, 64, 64 });

                // Заполняем тензор значениями из изображения
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        var pixel = image[x, y];
                        // Инвертируем: черный = 1.0, белый = 0.0
                        float value = 1.0f - (pixel.R / 255.0f);
                        tensor[0, 0, y, x] = value;
                    }
                }

                // Нормализуем как в ImageTransformer
                var mean = torch.tensor(new float[] { 0.5f }).reshape(1, 1, 1, 1);
                var std = torch.tensor(new float[] { 0.5f }).reshape(1, 1, 1, 1);

                return (tensor - mean) / std;
            }
        }

        private List<string> GetClassLabels()
        {
            return _dataLoader?.Dataset?.LabelToClass?
                .OrderBy(kv => kv.Key)
                .Select(kv => kv.Value)
                .ToList() ?? GetDefaultCyrillicLabels();
        }

        private List<string> GetDefaultCyrillicLabels()
        {
            // Резервный список кириллических букв (А-Я + Ё)
            var letters = new List<string>();

            // А-Е
            for (char c = 'А'; c <= 'Е'; c++)
                letters.Add(c.ToString());

            // Ё
            letters.Add("Ё");

            // Ж-Я  
            for (char c = 'Ж'; c <= 'Я'; c++)
                letters.Add(c.ToString());

            return letters;
        }

        private void DisplayRecognitionResults(List<(string letter, double confidence)> results)
        {
            if (results.Count == 0)
            {
                tbRecognitionResult.Text = "Буква не распознана";
                tbRecognitionConfidence.Text = "Уверенность: < 0.1%";
                icTopPredictions.ItemsSource = null;
            }
            else
            {
                var topResult = results.First();
                tbRecognitionResult.Text = $"Распознано: {topResult.letter}";
                tbRecognitionConfidence.Text = $"Уверенность: {topResult.confidence:F1}%";

                // Показываем топ-5 предсказаний
                var topPredictions = results.Select(r =>
                    $"{r.letter}: {r.confidence:F1}%").ToList();
                icTopPredictions.ItemsSource = topPredictions;
            }

            recognitionResultBorder.Visibility = Visibility.Visible;
        }

        private void btnExportResults_Click(object sender, RoutedEventArgs e)
        {

        }

        private void ExportResultsToFile(string filePath)
        {

        }
    }

    // Вспомогательные классы для Data Binding
    // Добавим в MainWindow.xaml.cs
    public class NASResult
    {
        public ConcreteArchitecture Architecture { get; set; }

        public double Accuracy { get; set; }
        public double TrainingTime { get; set; }
        public int Parameters { get; set; }
        public int Generation { get; set; }
        public string Algorithm { get; set; }
        public DateTime Timestamp { get; set; }
        public List<string> GeneticHistory { get; set; }
        public List<TrainingEpoch> TrainingHistory { get; set; }
        public DynamicCNN CNNModel { get; set; }

        public NASResult()
        {
            Timestamp = DateTime.Now;
            GeneticHistory = new List<string>();
        }

        // Конструктор из GeneticNASController.ArchitectureIndividual
        public NASResult(GeneticNASController.ArchitectureIndividual individual, string algorithm)
        {
            Architecture = individual.Architecture;
            Accuracy = individual.Accuracy;
            TrainingTime = individual.TrainingTime;
            Parameters = individual.Parameters;
            Generation = individual.Generation;
            Algorithm = algorithm;
            Timestamp = DateTime.Now;
            GeneticHistory = new List<string>(individual.GeneticHistory);
        }

        // Конструктор из RandomNASController.ArchitectureResult
        public NASResult(RandomNASController.ArchitectureResult result, string algorithm)
        {
            Architecture = result.Architecture;
            CNNModel = result.CNNModel;
            Accuracy = result.Accuracy;
            TrainingTime = result.TrainingTime;
            Parameters = result.Parameters;
            Generation = 0;
            Algorithm = algorithm;
            Timestamp = result.Timestamp;
            GeneticHistory = new List<string> { "Random Search" };
            TrainingHistory = result.TrainingHistory;
        }
    }

    public class ArchitectureViewModel
    {
        public string Name { get; set; }
        public int LayersCount { get; set; }
        public double Accuracy { get; set; }
        public int Parameters { get; set; }
        public double TrainingTime { get; set; }
        public int Generation { get; set; }
        public string Algorithm { get; set; }
        public string DisplayName { get; set; }
        public DynamicCNN CNNModel { get; set; }

        public ArchitectureViewModel(NASResult result)
        {
            Name = result.Architecture.Name;
            CNNModel = result.CNNModel;
            LayersCount = result.Architecture.Layers.Count;
            Accuracy = result.Accuracy;
            Parameters = result.Parameters;
            TrainingTime = result.TrainingTime;
            Generation = result.Generation;
            Algorithm = result.Algorithm;

            DisplayName = result.Algorithm == "Genetic"
                ? $"🧬 Gen{result.Generation}: {result.Architecture.Name}"
                : $"🎲 {result.Architecture.Name}";
        }
    }

    public class LayerInfo
    {
        public string Description { get; set; }
        public string Type { get; set; }
        public string ShortDescription { get; set; }

        public LayerInfo(Layer layer)
        {
            Description = layer.GetDescription();
            Type = layer.Type;
            ShortDescription = GetLayerShortDescription(layer);
        }

        private string GetLayerShortDescription(Layer layer)
        {
            return layer switch
            {
                ConvLayer conv => $"Conv {conv.Filters}f@{conv.KernelSize}x{conv.KernelSize}",
                PoolingLayer pool => $"{pool.PoolType} Pool {pool.PoolSize}x{pool.PoolSize}",
                FullyConnectedLayer fc => $"Dense {fc.Units} units",
                OutputLayer output => $"Output {output.NumClasses} classes",
                CustomLayer custom => $"{custom.Type}",
                _ => layer.GetType().Name
            };
        }
    }
    public class SearchConfig
    {
        public int NumTrials { get; set; }
        public int EpochsPerTrial { get; set; }
        public int MinLayers { get; set; }
        public int MaxLayers { get; set; }
        public int PopulationSize { get; set; }
        public int Generations { get; set; }
        public double MutationRate { get; set; }
        public double CrossoverRate { get; set; }
        public int BatchSize { get; set; }
        public int ImageSize { get; set; }
    }
}