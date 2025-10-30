using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using Курс.Core.NeuralNetworks;
using static TorchSharp.torch.optim;
using static Курс.NAS.Controllers.RandomNASController;
using Курс.NAS.Controllers;

namespace Курс.Core.Training
{
    public class ModelTrainer
    {
        private Device _device;
        private List<TrainingEpoch> _currentTrainingHistory;

        public ModelTrainer(Device device = null)
        {
            _device = device ?? (torch.cuda.is_available() ? CUDA : CPU);
        }

        public double TrainAndEvaluate(DynamicCNN model, PrecomputedBatches batches,
                                 int numEpochs, ArchitectureResult result = null,
                                 int patience = 20, double targetAccuracy = 99.0)
        {
            var optimizer = optim.Adam(model.parameters(), lr: 0.001);
            var criterion = nn.CrossEntropyLoss();
            //var scheduler = optim.lr_scheduler.StepLR(optimizer, step_size: 10, gamma: 0.1);

            double bestAccuracy = 0.0;
            int epochsWithoutImprovement = 0;
            int bestEpoch = 0;

            _currentTrainingHistory = new List<TrainingEpoch>();

            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                model.train();
                double trainLoss = 0.0;
                double trainAccuracy = 0.0;

                foreach (var batch in batches.TrainBatches)
                {
                    var output = model.forward(batch.Images);
                    var loss = criterion.forward(output, batch.Labels);

                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();

                    trainLoss += loss.item<float>();
                    trainAccuracy += CalculateAccuracy(output, batch.Labels);

                    output.Dispose();
                    loss.Dispose();
                }

                model.eval();
                double valAccuracy = 0.0;
                double valLoss = 0.0;

                using (no_grad())
                {
                    foreach (var batch in batches.ValBatches)
                    {
                        var valOutput = model.forward(batch.Images);
                        valAccuracy += CalculateAccuracy(valOutput, batch.Labels);
                        var loss = criterion.forward(valOutput, batch.Labels);
                        valLoss += loss.item<float>();
                        valOutput.Dispose();
                        loss.Dispose();
                    }
                }

                //scheduler.step();
                var currentLR = GetCurrentLearningRate(optimizer);

                double avgTrainLoss = trainLoss / batches.TrainBatches.Count;
                double avgValLoss = valLoss / batches.ValBatches.Count;
                double avgTrainAccuracy = trainAccuracy / batches.TrainBatches.Count;
                double avgValAccuracy = valAccuracy / batches.ValBatches.Count;

                var epochData = new TrainingEpoch
                {
                    Epoch = epoch + 1,
                    TrainLoss = avgTrainLoss,
                    ValLoss = avgValLoss,
                    TrainAccuracy = avgTrainAccuracy,
                    ValAccuracy = avgValAccuracy,
                    LearningRate = currentLR
                };

                _currentTrainingHistory.Add(epochData);

                result?.AddEpoch(epoch + 1, avgTrainLoss, avgValLoss, avgTrainAccuracy, avgValAccuracy, currentLR);

                if (avgValAccuracy > bestAccuracy)
                {
                    bestAccuracy = avgValAccuracy;
                    bestEpoch = epoch + 1;
                    epochsWithoutImprovement = 0;
                }
                else
                {
                    epochsWithoutImprovement++;
                }

                if (avgValAccuracy >= targetAccuracy || epochsWithoutImprovement >= patience)
                {
                    break;
                }
            }

            if (result != null)
            {
                result.TrainingHistory = _currentTrainingHistory;
            }

            optimizer.Dispose();
            criterion.Dispose();

            return bestAccuracy;
        }

        private double GetCurrentLearningRate(Optimizer optimizer)
        {
            try
            {
                var paramGroup = optimizer.ParamGroups.FirstOrDefault();
                return paramGroup?.LearningRate ?? 0.001;
            }
            catch
            {
                return 0.001;
            }
        }

        private double CalculateAccuracy(Tensor output, Tensor target)
        {
            var predictions = output.argmax(1);
            var correct = predictions.eq(target).sum().item<long>();
            var total = target.shape[0];
            return 100.0 * correct / total;
        }

        public List<TrainingEpoch> GetCurrentTrainingHistory()
        {
            return _currentTrainingHistory?.ToList() ?? new List<TrainingEpoch>();
        }
    }
}
