using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace Курс
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
            _device = device ?? (torch.cuda.is_available() ? CUDA : CPU);
            _results = new List<ArchitectureResult>();
        }

        public class ArchitectureResult
        {
            public Architecture Architecture { get; set; }
            public double Accuracy { get; set; }
            public double TrainingTime { get; set; }
            public int Parameters { get; set; }
            public DateTime Timestamp { get; set; }
        }

        public ArchitectureResult Search(CyrillicDataLoader dataLoader, int numTrials = 50,
                                       int minLayers = 5, int maxLayers = 12,
                                       int epochsPerTrial = 5, int batchSize = 32, int imageSize = 64)
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

                    // Обучение и оценка
                    var startTime = DateTime.Now;
                    var accuracy = _trainer.TrainAndEvaluate(model, batches, epochsPerTrial);
                    var trainingTime = (DateTime.Now - startTime).TotalSeconds;

                    // Сохранение результата
                    var result = new ArchitectureResult
                    {
                        Architecture = architecture.Clone(),
                        Accuracy = accuracy,
                        TrainingTime = trainingTime,
                        Parameters = (int)model.parameters().Sum(p => p.numel()),
                        Timestamp = DateTime.Now
                    };

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
}
