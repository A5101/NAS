using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using NAS.Data;
using NAS.NAS.Controllers;
using NAS.NAS.Models;

namespace NAS
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("ПОЛНЫЙ ЦИКЛ NAS С DYNAMICCNN");
            Console.WriteLine("=".PadRight(50, '='));

            try
            {
                Device device = cuda.is_available() ? CUDA : CPU;

                string dataPath = @"Cyrillic";
                int imageSize = 64;
                var dataLoader = new CyrillicDataLoader(dataPath, imageSize: imageSize);

                var nasController = new RandomNASController(
                                                            imageSize: imageSize,
                                                            device: device
                                                        );

                var bestArchitecture = nasController.Search(
                    dataLoader: dataLoader,
                    numTrials: 100,
                    minLayers: 5,
                    maxLayers: 20,
                    epochsPerTrial: 100,
                    batchSize: 32,
                     imageSize: imageSize
                );

                //var geneticConfig = new GeneticConfig
                //{
                //    PopulationSize = 15,
                //    Generations = 30,
                //    CrossoverRate = 0.7,
                //    MutationRate = 0.4,
                //    EliteRatio = 0.1,
                //    EpochsPerEvaluation = 15
                //};

                //var geneticController = new GeneticNASController(
                //    imageSize: imageSize,
                //    device: device,
                //    config: geneticConfig,
                //    seed: 42
                //);

                // Запуск эволюции
                //var bestArchitecture = geneticController.Evolve(
                //    dataLoader: dataLoader,
                //    batchSize: 32,
                //    imageSize: imageSize
                //);

            }
            catch (Exception ex)
            {
                Console.WriteLine($"КРИТИЧЕСКАЯ ОШИБКА: {ex.Message}");
                Console.WriteLine($"StackTrace: {ex.StackTrace}");
            }
        }
    }
}
