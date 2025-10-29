using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using Курс.Core.Architecture;
using Курс.Core.Training;
using Курс.Data;
using Курс.NAS.Generators;
using Курс.Core.NeuralNetworks;
using static Курс.NAS.Controllers.GeneticNASController;

namespace Курс.NAS.Controllers
{
    public class RandomNASController
    {
        private readonly ArchitectureGenerator _generator;
        private readonly ModelTrainer _trainer;
        private readonly Device _device;
        private List<ArchitectureResult> _results;

        public RandomNASController(int imageSize = 64, Device device = null, int? seed = null)
        {
            _generator = new ArchitectureGenerator(imageSize, seed);
            _trainer = new ModelTrainer(device);
            _device = device ?? (cuda.is_available() ? CUDA : CPU);
            _results = new List<ArchitectureResult>();
        }

        public class ArchitectureResult
        {
            public ConcreteArchitecture Architecture { get; set; }
            public double Accuracy { get; set; }
            public double TrainingTime { get; set; }
            public int Parameters { get; set; }
            public DateTime Timestamp { get; set; }

            // Новая коллекция для хранения истории обучения
            public List<TrainingEpoch> TrainingHistory { get; set; }

            public ArchitectureResult()
            {
                TrainingHistory = new List<TrainingEpoch>();
                Timestamp = DateTime.Now;
            }

            // Метод для добавления записи об эпохе
            public void AddEpoch(int epoch, double trainLoss, double valLoss, double trainAccuracy, double valAccuracy, double learningRate)
            {
                TrainingHistory.Add(new TrainingEpoch
                {
                    Epoch = epoch,
                    TrainLoss = trainLoss,
                    ValLoss = valLoss,
                    TrainAccuracy = trainAccuracy,
                    ValAccuracy = valAccuracy,
                    LearningRate = learningRate,
                    Timestamp = DateTime.Now
                });
            }

            // Метод для получения последних N эпох
            public List<TrainingEpoch> GetRecentEpochs(int count)
            {
                return TrainingHistory
                    .OrderByDescending(e => e.Epoch)
                    .Take(count)
                    .OrderBy(e => e.Epoch)
                    .ToList();
            }

            // Метод для проверки, была ли стагнация в обучении
            public bool HadTrainingPlateau(int windowSize = 10, double tolerance = 0.001)
            {
                if (TrainingHistory.Count < windowSize) return false;

                var recentEpochs = TrainingHistory
                    .OrderByDescending(e => e.Epoch)
                    .Take(windowSize)
                    .OrderBy(e => e.Epoch)
                    .ToList();

                var firstAccuracy = recentEpochs.First().ValAccuracy;
                var lastAccuracy = recentEpochs.Last().ValAccuracy;

                return Math.Abs(lastAccuracy - firstAccuracy) <= tolerance;
            }

            // Метод для получения лучшей эпохи по валидационной точности
            public TrainingEpoch GetBestEpoch()
            {
                return TrainingHistory
                    .OrderByDescending(e => e.ValAccuracy)
                    .ThenBy(e => e.ValLoss)
                    .FirstOrDefault();
            }

            // Метод для вычисления скорости обучения (accuracy/epoch)
            public double GetLearningSpeed()
            {
                if (TrainingHistory.Count < 2) return 0;

                var firstEpoch = TrainingHistory.OrderBy(e => e.Epoch).First();
                var lastEpoch = TrainingHistory.OrderByDescending(e => e.Epoch).First();

                var accuracyGain = lastEpoch.ValAccuracy - firstEpoch.ValAccuracy;
                var epochDifference = lastEpoch.Epoch - firstEpoch.Epoch;

                return epochDifference > 0 ? accuracyGain / epochDifference : 0;
            }
        }

        // Новый класс для хранения данных об одной эпохе обучения
    

        public ArchitectureResult Search(CyrillicDataLoader dataLoader, int numTrials = 50,
                                       int minLayers = 5, int maxLayers = 12,
                                       int epochsPerTrial = 5, int batchSize = 32, int imageSize = 64, IProgress<ArchitectureResult> progress = null)
        {
            Console.WriteLine($"ЗАПУСК СЛУЧАЙНОГО ПОИСКА АРХИТЕКТУР");
            Console.WriteLine($"   Trials: {numTrials}, Epochs per trial: {epochsPerTrial}");
            Console.WriteLine($"   Layers range: {minLayers}-{maxLayers}");
            Console.WriteLine($"   Устройство: {_device}");
            Console.WriteLine("=".PadRight(60, '='));

            ArchitectureResult bestResult = null;
            using var batches = dataLoader.PrecomputeBatches(batchSize, _device);
            for (int trial = 0; trial < numTrials; trial++)
            {
                Console.WriteLine($"\nТРИАЛ {trial + 1}/{numTrials}");

                try
                {
                    // Генерация случайной архитектуры
                    var architecture = _generator.GenerateRandomArchitecture(minLayers, maxLayers, dataLoader.Dataset.NumClasses);

                    Console.WriteLine(architecture.GetSummary());

                    // Создание модели
                    using var model = new DynamicCNN(architecture, inputChannels: 1, device: _device, inputHeight: imageSize, inputWidth: imageSize);
                    var result = new ArchitectureResult
                    {
                        Architecture = architecture.Clone(),
                        Parameters = (int)model.parameters().Sum(p => p.numel()),
                        Timestamp = DateTime.Now
                    };

                    // Обучение и оценка
                    var startTime = DateTime.Now;
                    var accuracy = _trainer.TrainAndEvaluate(model, batches, epochsPerTrial, result);
                    var trainingTime = (DateTime.Now - startTime).TotalSeconds;
                    result.Accuracy = accuracy;
                    result.TrainingTime = trainingTime;
                 
                    
                    progress?.Report(result);
                    _results.Add(result);

                    // Обновление лучшего результата
                    if (bestResult == null || accuracy > bestResult.Accuracy)
                    {
                        bestResult = result;
                        Console.WriteLine($"НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ: {accuracy:F2}%");
                    }

                    Console.WriteLine($"\nИТОГ ТРИАЛА: {accuracy:F2}% за {trainingTime:F1}с");

                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ошибка в триале {trial + 1}: {ex.Message}");
                }
            }

            PrintFinalResults(bestResult);
            return bestResult;
        }

        private void PrintFinalResults(ArchitectureResult bestResult)
        {
            Console.WriteLine($"\nСЛУЧАЙНЫЙ ПОИСК ЗАВЕРШЕН");
            Console.WriteLine("=".PadRight(60, '='));

            if (bestResult != null)
            {
                Console.WriteLine($"ЛУЧШАЯ АРХИТЕКТУРА:");
                Console.WriteLine($"   Точность: {bestResult.Accuracy:F2}%");
                Console.WriteLine($"   Время обучения: {bestResult.TrainingTime:F1}с");
                Console.WriteLine($"   Параметров: {bestResult.Parameters:N0}");
                Console.WriteLine($"   Слоев: {bestResult.Architecture.Layers.Count}");
                Console.WriteLine(bestResult.Architecture.GetSummary());
            }
        }
    }

    public class TrainingEpoch
    {
        public int Epoch { get; set; }
        public double TrainLoss { get; set; }
        public double ValLoss { get; set; }
        public double TrainAccuracy { get; set; }
        public double ValAccuracy { get; set; }
        public double LearningRate { get; set; }
        public DateTime Timestamp { get; set; }

        // Вычисляемые свойства
        public double LossDifference => TrainLoss - ValLoss;
        public double AccuracyDifference => ValAccuracy - TrainAccuracy;
        public bool IsOverfitting => LossDifference > 0.1 && AccuracyDifference < -2.0;

        public override string ToString()
        {
            return $"Epoch {Epoch}: Train={TrainLoss:F4}({TrainAccuracy:F2}%), Val={ValLoss:F4}({ValAccuracy:F2}%), LR={LearningRate:E2}";
        }
    }
}
