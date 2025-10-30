using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Курс.Core;
using Курс.Core.Architecture;

namespace Курс.NAS.Generators
{
    public class ArchitectureGenerator
    {
        private Random _random;
        private int _imageSize;

        public ArchitectureGenerator(int imageSize = 64, int? seed = null)
        {
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _imageSize = imageSize;
        }

        public ConcreteArchitecture GenerateArchitecture(int numLayers, int numClasses, string name = "GeneratedArch")
        {
            var architecture = new ConcreteArchitecture(name);

            if (numLayers < 3)
                throw new ArgumentException("Минимальное количество слоев: 3 (conv + pool + dense)");

            int numDenseLayers = Random.Shared.Next(1, (int)Math.Floor(numLayers / 3.0));
            int numConvPoolPairs = (numLayers - numDenseLayers) / 2;

            Console.WriteLine($"Генерация архитектуры: {numConvPoolPairs} conv-pool пар, {numDenseLayers} dense слоев");

            GenerateConvolutionalLayers(architecture, numConvPoolPairs);

            architecture.AddLayer(new CustomLayer("flatten", "flatten", "none"));

            GenerateFullyConnectedLayers(architecture, numDenseLayers);

            architecture.AddLayer(new OutputLayer("output", numClasses));

            if (!architecture.Validate())
            {
                throw new InvalidOperationException("Сгенерированная архитектура невалидна");
            }
            architecture.Name = $"RandomArch_{architecture.Layers.Count}L";
            return architecture;
        }

        private void GenerateConvolutionalLayers(ConcreteArchitecture architecture, int numPairs)
        {
            int size = _imageSize;

            int[] filters = { 1, 2, 4, 8, 16 };

            int filtersCount = filters[Random.Shared.Next(0, filters.Length)];

            for (int i = 0; i < numPairs; i++)
            {
                var convLayer = CreateConvLayer($"conv_{i + 1}", filtersCount);
                size = size - convLayer.KernelSize + 1;

                var poolLayer = CreatePoolLayer($"pool_{i + 1}");
                size /= poolLayer.PoolSize;
                if (size < 4)
                    return;

                architecture.AddLayer(convLayer);
                architecture.AddLayer(poolLayer);

                filtersCount = GetNextFilters(filtersCount, i);
            }
        }

        private ConvLayer CreateConvLayer(string name, int filters)
        {
            var kernelSize = _random.Next(2) == 0 ? 3 : 5; // 3x3 или 5x5
            var activation = _random.Next(2) == 0 ? "relu" : "leaky_relu";
            var useBatchNorm = _random.Next(2) == 0;

            return new ConvLayer(name, filters, kernelSize, activation, useBatchNorm: useBatchNorm);
        }

        private PoolingLayer CreatePoolLayer(string name)
        {
            var poolType = _random.Next(2) == 0 ? "max" : "avg";
            return new PoolingLayer(name, poolType, poolSize: 2, stride: 2);
        }

        private int GetNextFilters(int currentFilters, int layerIndex)
        {
            int[] multipliers = { 1, 2, 4, 8 };
            int nextFilters = currentFilters * multipliers[Random.Shared.Next(0, multipliers.Length)];
            return Math.Min(nextFilters, 512);
        }

        private void GenerateFullyConnectedLayers(ConcreteArchitecture architecture, int numDenseLayers)
        {
            if (numDenseLayers <= 0) return;

            long inputUnits = CalculateFlattenSize(architecture);

            Console.WriteLine($"   Размер flatten: {inputUnits} нейронов");

            for (int i = 0; i < numDenseLayers; i++)
            {
                var units = CalculateDenseUnits(inputUnits, i, numDenseLayers);
                var dropout = CalculateDropoutRate(i, numDenseLayers);
                var activation = i == numDenseLayers - 1 ? "none" : "relu"; 

                var denseLayer = new FullyConnectedLayer(
                    $"dense_{i + 1}",
                    (int)units,
                    activation,
                    dropout,
                    useBatchNorm: false
                );

                architecture.AddLayer(denseLayer);
                inputUnits = units; 

                Console.WriteLine($"   Dense #{i + 1}: {units} нейронов, dropout: {dropout:F2}");
            }
        }

        private long CalculateFlattenSize(ConcreteArchitecture architecture)
        {
            var finalSize = architecture.CalculateFinalSize(1, _imageSize, _imageSize);

            long flattenSize = finalSize.channels * finalSize.height * finalSize.width;

            return flattenSize;
        }

        private long CalculateDenseUnits(long inputUnits, int denseIndex, int totalDenseLayers)
        {
            if (denseIndex == 0)
            {
                return Math.Min(inputUnits, 256);
            }

            double reductionFactor = (totalDenseLayers - denseIndex) / (double)totalDenseLayers;
            long units = (long)(inputUnits * reductionFactor);

            return Math.Max(units, 32);
        }

        private double CalculateDropoutRate(int denseIndex, int totalDenseLayers)
        {
            double baseRate = 0.2;
            double increment = 0.15;

            return Math.Min(baseRate + denseIndex * increment, 0.5);
        }

        public ConcreteArchitecture GenerateRandomArchitecture(int minLayers = 4, int maxLayers = 10, int numClasses = 33)
        {
            int numLayers = _random.Next(minLayers, maxLayers + 1);
            return GenerateArchitecture(numLayers, numClasses, $"RandomArch_{numLayers}L");
        }
    }
}
