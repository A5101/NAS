using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using Курс.Core.Architecture;
using Курс.NAS.Models;
using Курс.Core.Training;
using Курс.NAS.Generators;
using Курс.Core.NeuralNetworks;
using Курс.Data;
using System.Diagnostics;

namespace Курс.NAS.Controllers
{
    public class GeneticNASController
    {
        private readonly ArchitectureGenerator _generator;
        private readonly ModelTrainer _trainer;
        private readonly Device _device;
        private List<ArchitectureIndividual> _population;
        private readonly GeneticConfig _config;
        private Random _random;

        public GeneticNASController(int imageSize = 64, Device device = null,
                                  GeneticConfig config = null, int? seed = null)
        {
            _generator = new ArchitectureGenerator(imageSize, seed);
            _trainer = new ModelTrainer(device);
            _device = device ?? (torch.cuda.is_available() ? CUDA : CPU);
            _config = config ?? new GeneticConfig();
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _population = new List<ArchitectureIndividual>();
        }

        

        public class ArchitectureIndividual
        {
            public ConcreteArchitecture Architecture { get; set; }
            public double Fitness { get; set; }
            public double Accuracy { get; set; }
            public double TrainingTime { get; set; }
            public int Parameters { get; set; }
            public int Generation { get; set; }
            public List<string> GeneticHistory { get; set; }

            public ArchitectureIndividual()
            {
                GeneticHistory = new List<string>();
            }

            public override string ToString()
            {
                return $"Gen{Generation}: Fit={Fitness:F2}, Acc={Accuracy:F2}%, Params={Parameters:N0}";
            }
        }

        public ArchitectureIndividual Evolve(CyrillicDataLoader dataLoader, int batchSize = 32, int imageSize = 64, IProgress<ArchitectureIndividual> progress = null)
        {
            Console.WriteLine($"ЗАПУСК ГЕНЕТИЧЕСКОГО ПОИСКА АРХИТЕКТУР");
            Console.WriteLine($"   Популяция: {_config.PopulationSize}, Поколений: {_config.Generations}");
            Console.WriteLine($"   Crossover: {_config.CrossoverRate}, Mutation: {_config.MutationRate}");
            Console.WriteLine($"   Elite: {_config.EliteRatio * 100}%, Tournament: {_config.TournamentSize}");
            Console.WriteLine("=".PadRight(70, '='));

            // 1. Инициализация популяции
            InitializePopulation(dataLoader.Dataset.NumClasses);

            ArchitectureIndividual bestIndividual = null;
            using var batches = dataLoader.PrecomputeBatches(batchSize, _device);

            for (int generation = 0; generation < _config.Generations; generation++)
            {
                Console.WriteLine($"\nПОКОЛЕНИЕ {generation + 1}/{_config.Generations}");

                // 2. Оценка приспособленности
                EvaluatePopulation(batches, generation, imageSize);

                // 3. Находим лучшую особь
                var currentBest = _population.OrderByDescending(ind => ind.Fitness).First();
                progress?.Report(currentBest);
                if (bestIndividual == null || currentBest.Fitness > bestIndividual.Fitness)
                {
                    bestIndividual = currentBest;
                    Console.WriteLine($"НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ: {bestIndividual}");
                }

                // 4. Вывод статистики поколения
                PrintGenerationStats(generation);

                // 5. Создание нового поколения
                if (generation < _config.Generations - 1) // Не создаем новое поколение для последней итерации
                {
                    var newPopulation = CreateNewGeneration();
                    _population = newPopulation;
                }

                // 6. Проверка критерия остановки (ранняя остановка)
                if (ShouldEarlyStop(generation))
                {
                    Console.WriteLine($"РАННЯЯ ОСТАНОВКА на поколении {generation + 1}");
                    break;
                }
            }

            PrintFinalResults(bestIndividual);
            return bestIndividual;
        }

        private void InitializePopulation(int numClasses)
        {
            Console.WriteLine($"Инициализация популяции из {_config.PopulationSize} особей...");

            for (int i = 0; i < _config.PopulationSize; i++)
            {
                var numLayers = _random.Next(_config.MinLayers, _config.MaxLayers + 1);
                var architecture = _generator.GenerateArchitecture(numLayers, numClasses, $"Gen0_Ind{i}");

                _population.Add(new ArchitectureIndividual
                {
                    Architecture = architecture,
                    Generation = 0,
                    GeneticHistory = new List<string> { "Initialization" }
                });
            }
        }

        private void EvaluatePopulation(PrecomputedBatches batches, int generation, int imageSize)
        {
            int evaluated = 0;
            foreach (var individual in _population)
            {
                if (individual.Fitness > 0) continue; // Уже оценен

                try
                {
                    evaluated++;
                    Console.WriteLine($"   Оценка особи {evaluated}/{_population.Count}...");

                    using var model = new DynamicCNN(individual.Architecture, inputChannels: 1,
                                                   inputHeight: imageSize, inputWidth: imageSize, device: _device);

                    var startTime = DateTime.Now;
                    var accuracy = _trainer.TrainAndEvaluate(model, batches, _config.EpochsPerEvaluation);
                    var trainingTime = (DateTime.Now - startTime).TotalSeconds;

                    individual.Accuracy = accuracy;
                    individual.TrainingTime = trainingTime;
                    individual.Fitness = CalculateFitness(individual);
                    individual.Generation = generation;

                    Console.WriteLine($"     Accuracy: {accuracy:F2}%, Fitness: {individual.Fitness:F2}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"     Ошибка оценки: {ex.Message}");
                    individual.Fitness = 0.01; // Минимальная приспособленность
                    individual.Accuracy = 0;
                }
            }
        }

        private double CalculateFitness(ArchitectureIndividual individual)
        {
            // Multi-objective fitness функция
            double accuracyScore = individual.Accuracy / 100.0; // Нормализуем к [0, 1]
            double complexityPenalty = 1.0 / (1.0 + Math.Log(individual.Parameters + 1) / 10.0);
            double speedBonus = 1.0 / (1.0 + individual.TrainingTime / 60.0); // Нормализуем время

            return accuracyScore * 0.7 + complexityPenalty * 0.2 + speedBonus * 0.1;
        }

        private List<ArchitectureIndividual> CreateNewGeneration()
        {
            var newPopulation = new List<ArchitectureIndividual>();

            // 1. Элитизм - сохраняем лучших особей
            var eliteCount = (int)(_config.PopulationSize * _config.EliteRatio);
            var elites = _population.OrderByDescending(ind => ind.Fitness).Take(eliteCount).ToList();

            foreach (var elite in elites)
            {
                newPopulation.Add(new ArchitectureIndividual
                {
                    Architecture = elite.Architecture.Clone(),
                    Fitness = elite.Fitness,
                    Accuracy = elite.Accuracy,
                    TrainingTime = elite.TrainingTime,
                    Parameters = elite.Parameters,
                    Generation = elite.Generation + 1,
                    GeneticHistory = new List<string>(elite.GeneticHistory) { "Elite" }
                });
            }

            // 2. Скрещивание и мутация для остальной части популяции
            while (newPopulation.Count < _config.PopulationSize)
            {
                if (_random.NextDouble() < _config.CrossoverRate && newPopulation.Count < _config.PopulationSize - 1)
                {
                    // Кроссовер - создаем двух потомков
                    var parent1 = TournamentSelection();
                    var parent2 = TournamentSelection();

                    var (child1, child2) = Crossover(parent1, parent2);
                    newPopulation.Add(child1);

                    if (newPopulation.Count < _config.PopulationSize)
                        newPopulation.Add(child2);
                }
                else
                {
                    // Мутация
                    var parent = TournamentSelection();
                    var child = Mutate(parent);
                    newPopulation.Add(child);
                }
            }

            return newPopulation;
        }

        private ArchitectureIndividual TournamentSelection()
        {
            var tournament = _population.OrderBy(x => _random.Next()).Take(_config.TournamentSize).ToList();
            return tournament.OrderByDescending(ind => ind.Fitness).First();
        }

        private (ArchitectureIndividual, ArchitectureIndividual) Crossover(ArchitectureIndividual parent1, ArchitectureIndividual parent2)
        {
            var arch1 = parent1.Architecture.Clone();
            var arch2 = parent2.Architecture.Clone();

            // Одноточечный кроссовер
            var crossoverPoint = _random.Next(1, Math.Min(arch1.Layers.Count, arch2.Layers.Count) - 1);

            var child1Layers = new List<Layer>();
            var child2Layers = new List<Layer>();

            // Первая часть от parent1, вторая от parent2
            child1Layers.AddRange(arch1.Layers.Take(crossoverPoint));
            child1Layers.AddRange(arch2.Layers.Skip(crossoverPoint));

            // Первая часть от parent2, вторая от parent1  
            child2Layers.AddRange(arch2.Layers.Take(crossoverPoint));
            child2Layers.AddRange(arch1.Layers.Skip(crossoverPoint));

            arch1.Layers = child1Layers;
            arch2.Layers = child2Layers;

            // Валидация и исправление архитектур
            ValidateAndRepairArchitecture(arch1);
            ValidateAndRepairArchitecture(arch2);

            return (
                new ArchitectureIndividual
                {
                    Architecture = arch1,
                    Generation = Math.Max(parent1.Generation, parent2.Generation) + 1,
                    GeneticHistory = new List<string> { $"Crossover(P1:{parent1.Generation},P2:{parent2.Generation})" }
                },
                new ArchitectureIndividual
                {
                    Architecture = arch2,
                    Generation = Math.Max(parent1.Generation, parent2.Generation) + 1,
                    GeneticHistory = new List<string> { $"Crossover(P1:{parent1.Generation},P2:{parent2.Generation})" }
                }
            );
        }

        private ArchitectureIndividual Mutate(ArchitectureIndividual parent)
        {
            var childArch = parent.Architecture.Clone();
            var mutationType = _random.Next(4); // 4 типа мутации

            switch (mutationType)
            {
                case 0: // Добавление слоя
                    if (childArch.Layers.Count < _config.MaxLayers)
                    {
                        var insertPos = _random.Next(1, childArch.Layers.Count - 1);
                        var newLayer = GenerateRandomLayer("conv");
                        childArch.Layers.Insert(insertPos, newLayer);
                    }
                    break;

                case 1: // Удаление слоя
                    if (childArch.Layers.Count > _config.MinLayers)
                    {
                        var removePos = _random.Next(1, childArch.Layers.Count - 2); // Не удаляем первый и последний слои
                        childArch.Layers.RemoveAt(removePos);
                    }
                    break;

                case 2: // Изменение параметров
                    var changePos = _random.Next(childArch.Layers.Count - 1);
                    MutateLayerParameters(childArch.Layers[changePos]);
                    break;

                case 3: // Замена типа слоя
                    var replacePos = _random.Next(1, childArch.Layers.Count - 2);
                    var newLayerType = GenerateRandomLayer("any");
                    childArch.Layers[replacePos] = newLayerType;
                    break;
            }

            ValidateAndRepairArchitecture(childArch);

            return new ArchitectureIndividual
            {
                Architecture = childArch,
                Generation = parent.Generation + 1,
                GeneticHistory = new List<string>(parent.GeneticHistory) { $"Mutation(type:{mutationType})" }
            };
        }

        private Layer GenerateRandomLayer(string preferredType = "any")
        {
            var types = new[] { "conv", "pool", "dense" };
            var type = preferredType == "any" ? types[_random.Next(types.Length)] : preferredType;

            return type switch
            {
                "conv" => new ConvLayer($"mutated_conv", _random.Next(8, 65), _random.Next(3, 6)),
                "pool" => new PoolingLayer($"mutated_pool", _random.Next(2) == 0 ? "max" : "avg"),
                "dense" => new FullyConnectedLayer($"mutated_dense", _random.Next(32, 257)),
                _ => new ConvLayer($"mutated_conv", 32, 3)
            };
        }

        private void MutateLayerParameters(Layer layer)
        {
            switch (layer)
            {
                case ConvLayer conv:
                    conv.Filters = Math.Max(8, conv.Filters + _random.Next(-8, 9));
                    break;
                case FullyConnectedLayer dense:
                    dense.Units = Math.Max(16, dense.Units + _random.Next(-16, 17));
                    break;
                case PoolingLayer pool:
                    pool.PoolSize = _random.Next(2, 4);
                    break;
            }
        }

        private void ValidateAndRepairArchitecture(ConcreteArchitecture arch)
        {
            // Убеждаемся, что есть хотя бы один сверточный и выходной слой
            if (!arch.Layers.Any(l => l.Type == "conv"))
            {
                arch.Layers.Insert(1, GenerateRandomLayer("conv"));
            }

            if (!arch.Layers.Any(l => l.Type == "output"))
            {
                // Находим и заменяем последний dense слой на output
                var lastDense = arch.Layers.LastOrDefault(l => l.Type == "dense");
                if (lastDense != null)
                {
                    var outputLayer = new OutputLayer("output", ((FullyConnectedLayer)lastDense).Units);
                    arch.Layers[arch.Layers.IndexOf(lastDense)] = outputLayer;
                }
            }

            // Убеждаемся, что flatten есть перед dense слоями
            var firstDenseIndex = arch.Layers.FindIndex(l => l.Type == "dense");
            if (firstDenseIndex >= 0)
            {
                var hasFlatten = arch.Layers.Take(firstDenseIndex).Any(l => l.Type == "flatten");
                if (!hasFlatten)
                {
                    arch.Layers.Insert(firstDenseIndex, new CustomLayer("flatten", "flatten"));
                }
            }
        }

        private void PrintGenerationStats(int generation)
        {
            var stats = _population.OrderByDescending(ind => ind.Fitness).ToList();
            var best = stats.First();
            var worst = stats.Last();
            var avgFitness = stats.Average(ind => ind.Fitness);
            var avgAccuracy = stats.Average(ind => ind.Accuracy);

            Console.WriteLine($"СТАТИСТИКА ПОКОЛЕНИЯ {generation + 1}:");
            Console.WriteLine($"   Лучший: {best.Fitness:F2} fit, {best.Accuracy:F2}% acc");
            Console.WriteLine($"   Худший: {worst.Fitness:F2} fit, {worst.Accuracy:F2}% acc");
            Console.WriteLine($"   Средний: {avgFitness:F2} fit, {avgAccuracy:F2}% acc");
            Console.WriteLine($"   Размеры: {stats.Min(ind => ind.Architecture.Layers.Count)}-" +
                            $"{stats.Max(ind => ind.Architecture.Layers.Count)} слоев");
        }

        private bool ShouldEarlyStop(int generation)
        {
            if (generation < 10) return false; // Минимум 10 поколений

            // Проверяем, улучшалась ли лучшая приспособленность в последних 5 поколениях
            var recentBest = _population.Where(ind => ind.Generation >= generation - 5)
                                      .Max(ind => ind.Fitness);
            var currentBest = _population.Max(ind => ind.Fitness);

            return currentBest <= recentBest;
        }

        private void PrintFinalResults(ArchitectureIndividual bestIndividual)
        {
            Console.WriteLine($"\nГЕНЕТИЧЕСКИЙ ПОИСК ЗАВЕРШЕН");
            Console.WriteLine("=".PadRight(70, '='));

            if (bestIndividual != null)
            {
                Console.WriteLine($"ЛУЧШАЯ АРХИТЕКТУРА (Поколение {bestIndividual.Generation}):");
                Console.WriteLine($"   Fitness: {bestIndividual.Fitness:F2}");
                Console.WriteLine($"   Accuracy: {bestIndividual.Accuracy:F2}%");
                Console.WriteLine($"   Training Time: {bestIndividual.TrainingTime:F1}с");
                Console.WriteLine($"   Parameters: {bestIndividual.Parameters:N0}");
                Console.WriteLine($"   Layers: {bestIndividual.Architecture.Layers.Count}");
                Console.WriteLine($"   Genetic History: {string.Join(" → ", bestIndividual.GeneticHistory)}");

                Console.WriteLine(bestIndividual.Architecture.GetSummary());
            }

            // Вывод топ-5 архитектур
            var top5 = _population.OrderByDescending(ind => ind.Fitness).Take(5).ToList();
            Console.WriteLine($"\nТОП-5 АРХИТЕКТУР:");
            for (int i = 0; i < top5.Count; i++)
            {
                Console.WriteLine($"   {i + 1}. {top5[i]}");
            }
        }
    }
}
