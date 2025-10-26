using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace Курс
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("ПОЛНЫЙ ЦИКЛ NAS С DYNAMICCNN");
            Console.WriteLine("=".PadRight(50, '='));

            try
            {
                Device device = CPU;//cuda.is_available() ? CUDA : CPU;
                // 1. Загрузка данных
                string dataPath = @"Cyrillic";
                int imageSize = 32;
                var dataLoader = new CyrillicDataLoader(dataPath, imageSize: imageSize);

                var nasController = new RandomNASController(
                                                            imageSize: imageSize,
                                                            device: device
                                                        );

                // Запуск поиска
                var bestArchitecture = nasController.Search(
                    dataLoader: dataLoader,
                    numTrials: 100,
                    minLayers: 5,    
                    maxLayers: 20,     
                    epochsPerTrial: 100,     
                    batchSize: 32,
                     imageSize: imageSize
                );

                //// 2. Генерация архитектуры
                //var generator = new ArchitectureGenerator(imageSize: 64, seed: 42);
                //var architecture = generator.GenerateRandomArchitecture(minLayers: 6, maxLayers: 8, numClasses: dataLoader.Dataset.NumClasses);

                //Console.WriteLine(architecture.GetSummary());

                //// 3. Проверка совместимости
                //if (!architecture.CheckLayerCompatibility(channels:1, height:64, width:64))
                //{
                //    Console.WriteLine("Архитектура несовместима!");
                //    return;
                //}

                //// 4. Создание и тестирование модели
                //
                //var model = new DynamicCNN(architecture, device: device);
                //Console.WriteLine(model.GetModelInfo());
                //using PrecomputedBatches batches = dataLoader.PrecomputeBatches(batchSize: 32, device: device);
                //// 5. Тест прямого прохода
                //model.TestForwardPass();
                //var t1 = DateTime.Now;
                //// 6. Оценка архитектуры
                //var trainer = new ModelTrainer(device);
                //double accuracy = trainer.TrainAndEvaluate(model, batches, numEpochs: 500);
                //Console.WriteLine(DateTime.Now.Subtract(t1));

                //Console.WriteLine($"\nФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {accuracy:F2}%");
                //Console.WriteLine(architecture.GetSummary());

                //// Очистка
                //model.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"КРИТИЧЕСКАЯ ОШИБКА: {ex.Message}");
                Console.WriteLine($"StackTrace: {ex.StackTrace}");
            }
        }
    }
}
