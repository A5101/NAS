using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using NAS.Core.Architecture;
using NAS.Core.Training;
using NAS.Data;
using NAS.NAS.Generators;
using NAS.Core.NeuralNetworks;
using static NAS.NAS.Controllers.GeneticNASController;
using NAS.Core.Training;

namespace NAS.NAS.Controllers
{
    public class RandomNASController
    {
        private readonly ArchitectureGenerator _generator;
        private readonly ModelTrainer _trainer;
        private readonly Device _device;
        private List<ArchitectureResult> _results;
        private HashSet<string> _testedArchitectureHashes;

        public RandomNASController(int imageSize = 64, Device device = null, int? seed = null)
        {
            _generator = new ArchitectureGenerator(imageSize, seed);
            _trainer = new ModelTrainer(device);
            _device = device ?? (cuda.is_available() ? CUDA : CPU);
            _results = new List<ArchitectureResult>();
            _testedArchitectureHashes = new HashSet<string>();
        }

        public class ArchitectureResult
        {
            public ConcreteArchitecture Architecture { get; set; }
            public CNNModel CNNModel { get; set; }
            public double Accuracy { get; set; }
            public double TrainingTime { get; set; }
            public int Parameters { get; set; }
            public DateTime Timestamp { get; set; }
            public List<TrainingEpoch> TrainingHistory { get; set; }

            public ArchitectureResult()
            {
                TrainingHistory = new List<TrainingEpoch>();
                Timestamp = DateTime.Now;
            }

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

            public List<TrainingEpoch> GetRecentEpochs(int count)
            {
                return TrainingHistory
                    .OrderByDescending(e => e.Epoch)
                    .Take(count)
                    .OrderBy(e => e.Epoch)
                    .ToList();
            }

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

            public TrainingEpoch GetBestEpoch()
            {
                return TrainingHistory
                    .OrderByDescending(e => e.ValAccuracy)
                    .ThenBy(e => e.ValLoss)
                    .FirstOrDefault();
            }

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
                    ConcreteArchitecture architecture;
                    bool isUnique;
                    int attempts = 0;
                    const int maxAttempts = 50; 

                    do
                    {
                        architecture = _generator.GenerateRandomArchitecture(minLayers, maxLayers, dataLoader.Dataset.NumClasses);
                        isUnique = !IsArchitectureAlreadyTested(architecture);
                        attempts++;

                        if (attempts >= maxAttempts)
                        {
                            Console.WriteLine($"Достигнут лимит попыток генерации уникальной архитектуры ({maxAttempts})");
                            break;
                        }
                    }
                    while (!isUnique && attempts < maxAttempts);

                    if (!isUnique)
                    {
                        Console.WriteLine($"Пропуск дубликата архитектуры: {architecture.Name}");
                        continue; 
                    }

                    Console.WriteLine(architecture.GetSummary());

                    using var model = new DynamicCNN(architecture, inputChannels: 1, device: _device, inputHeight: imageSize, inputWidth: imageSize);
                    
                    var result = new ArchitectureResult
                    {
                        Architecture = architecture.Clone(),
                        Parameters = (int)model.parameters().Sum(p => p.numel()),
                        Timestamp = DateTime.Now
                    };

                    var startTime = DateTime.Now;
                    var accuracy = _trainer.TrainAndEvaluate(model, batches, epochsPerTrial, result);
                    var trainingTime = (DateTime.Now - startTime).TotalSeconds;
                    result.Accuracy = accuracy;
                    result.TrainingTime = trainingTime;

                    AddArchitectureToTested(architecture);
                    var modelData = model.Save();
                    result.CNNModel = modelData;
                    progress?.Report(result);
                    _results.Add(result);

                    if (bestResult == null || accuracy > bestResult.Accuracy)
                    {
                        bestResult = result;
                        Console.WriteLine($"НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ: {accuracy:F2}%");
                    }

                    Console.WriteLine($"\nИТОГ ТРИАЛА {trial + 1}: {accuracy:F2}% за {trainingTime:F1}с (уникальная архитектура)");

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

        private bool IsArchitectureAlreadyTested(ConcreteArchitecture architecture)
        {
            string architectureHash = CalculateArchitectureSignature(architecture);
            return _testedArchitectureHashes.Contains(architectureHash);
        }

        private void AddArchitectureToTested(ConcreteArchitecture architecture)
        {
            string architectureHash = CalculateArchitectureSignature(architecture);
            _testedArchitectureHashes.Add(architectureHash);
        }

        private string CalculateArchitectureHash(ConcreteArchitecture architecture)
        {
            var sb = new StringBuilder();

            sb.Append(architecture.Layers.Count);
            sb.Append("|");

            foreach (var layer in architecture.Layers)
            {
                sb.Append(layer.Type);
                sb.Append("|");

                switch (layer)
                {
                    case ConvLayer conv:
                        sb.Append(conv.Filters);
                        sb.Append("|");
                        sb.Append(conv.KernelSize);
                        sb.Append("|");
                        sb.Append(conv.Activation);
                        sb.Append("|");
                        break;

                    case PoolingLayer pool:
                        sb.Append(pool.PoolType);
                        sb.Append("|");
                        sb.Append(pool.PoolSize);
                        sb.Append("|");
                        break;

                    case FullyConnectedLayer fc:
                        sb.Append(fc.Units);
                        sb.Append("|");
                        sb.Append(fc.Activation);
                        sb.Append("|");
                        sb.Append(fc.DropoutRate.ToString("F2"));
                        sb.Append("|");
                        break;

                    case OutputLayer output:
                        sb.Append(output.NumClasses);
                        sb.Append("|");
                        break;

                    case CustomLayer custom:
                        sb.Append(custom.Type);
                        sb.Append("|");
                        break;
                }
            }

            return sb.ToString();
        }

        private string CalculateArchitectureSignature(ConcreteArchitecture architecture)
        {
            var signature = new List<string>
        {
            $"Layers:{architecture.Layers.Count}"
        };

            foreach (var layer in architecture.Layers)
            {
                string layerSignature = layer switch
                {
                    ConvLayer conv => $"CONV[{conv.Filters},{conv.KernelSize},{conv.Activation}]",
                    PoolingLayer pool => $"POOL[{pool.PoolType},{pool.PoolSize}]",
                    FullyConnectedLayer fc => $"DENSE[{fc.Units},{fc.Activation}]",
                    OutputLayer output => $"OUTPUT[{output.NumClasses}]",
                    CustomLayer custom => $"CUSTOM[{custom.Type}]",
                    _ => $"UNKNOWN[{layer.GetType().Name}]"
                };
                signature.Add(layerSignature);
            }

            return string.Join("|", signature);
        }
    }

   
}
