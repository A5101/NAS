using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace Курс
{
    public class ModelTrainer
    {
        private Device _device;

        public ModelTrainer(Device device = null)
        {
            _device = device ?? (torch.cuda.is_available() ? CUDA : CPU);
        }

        public double TrainAndEvaluate(DynamicCNN model, PrecomputedBatches batches,
                                 int numEpochs, int patience = 20, double targetAccuracy = 99.0)
        {
            var optimizer = optim.Adam(model.parameters(), lr: 0.001);
            var criterion = nn.CrossEntropyLoss();

            double bestAccuracy = 0.0;
            int epochsWithoutImprovement = 0;
            int bestEpoch = 0;

            // ДЛЯ АНАЛИЗА ПРОГРЕССА
            var epochHistory = new List<(int epoch, double trainLoss, double valAccuracy)>();

            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                // ОБУЧЕНИЕ
                model.train();
                double trainLoss = 0.0;

                foreach (var batch in batches.TrainBatches)
                {
                    var output = model.forward(batch.Images);
                    var loss = criterion.forward(output, batch.Labels);

                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();

                    trainLoss += loss.item<float>();

                    output.Dispose();
                    loss.Dispose();
                }

                // ВАЛИДАЦИЯ
                model.eval();
                double valAccuracy = 0.0;

                using (no_grad())
                {
                    foreach (var batch in batches.ValBatches)
                    {
                        var valOutput = model.forward(batch.Images);
                        valAccuracy += CalculateAccuracy(valOutput, batch.Labels);
                        valOutput.Dispose();
                    }
                }

                double avgTrainLoss = trainLoss / batches.TrainBatches.Count;
                double avgValAccuracy = valAccuracy / batches.ValBatches.Count;

                // СОХРАНЯЕМ ИСТОРИЮ
                epochHistory.Add((epoch + 1, avgTrainLoss, avgValAccuracy));

                // ОБНОВЛЯЕМ ЛУЧШИЙ РЕЗУЛЬТАТ
                bool improved = false;
                if (avgValAccuracy > bestAccuracy)
                {
                    double improvement = avgValAccuracy - bestAccuracy;
                    bestAccuracy = avgValAccuracy;
                    bestEpoch = epoch + 1;
                    epochsWithoutImprovement = 0;
                    improved = true;

                    Console.WriteLine($"Epoch {epoch + 1:00}/{numEpochs}:");
                    Console.WriteLine($"   Train Loss: {avgTrainLoss:F4}");
                    Console.WriteLine($"   Val Accuracy: {avgValAccuracy:F2}% 🆕 (+{improvement:F2}%)");
                    Console.WriteLine($"   Best Accuracy: {bestAccuracy:F2}%");
                }
                else
                {
                    epochsWithoutImprovement++;
                    Console.WriteLine($"Epoch {epoch + 1:00}/{numEpochs}:");
                    Console.WriteLine($"   Train Loss: {avgTrainLoss:F4}");
                    Console.WriteLine($"   Val Accuracy: {avgValAccuracy:F2}%");
                    Console.WriteLine($"   Best Accuracy: {bestAccuracy:F2}% (эпох без улучшений: {epochsWithoutImprovement}/{patience})");
                }

                // ПРОВЕРКА УСЛОВИЙ ОСТАНОВКИ
                if (avgValAccuracy >= targetAccuracy)
                {
                    Console.WriteLine($"\nЦЕЛЬ ДОСТИГНУТА! Точность {avgValAccuracy:F2}% >= {targetAccuracy}%");
                    break;
                }

                if (epochsWithoutImprovement >= patience)
                {
                    Console.WriteLine($"\nEARLY STOPPING! Нет улучшений {patience} эпох подряд");
                    Console.WriteLine($"   Лучшая точность: {bestAccuracy:F2}% на эпохе {bestEpoch}");
                    break;
                }

                if (epoch == numEpochs - 1)
                {
                    Console.WriteLine($"\nДОСТИГНУТО МАКСИМАЛЬНОЕ ЧИСЛО ЭПОХ");
                    Console.WriteLine($"   Лучшая точность: {bestAccuracy:F2}% на эпохе {bestEpoch}");
                }
            }

            optimizer.Dispose();
            criterion.Dispose();

            return bestAccuracy;
        }


        private double CalculateAccuracy(Tensor output, Tensor target)
        {
            var predictions = output.argmax(1);
            var correct = predictions.eq(target).sum().item<long>();
            var total = target.shape[0];
            return 100.0 * correct / total;
        }
    }
}
