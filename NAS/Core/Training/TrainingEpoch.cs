using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NAS.Core.Training
{
    public class TrainingEpoch
    {
        public int Epoch { get; set; }
        public double TrainLoss { get; set; }
        public double ValLoss { get; set; }
        public double TrainAccuracy { get; set; }
        public double ValAccuracy { get; set; }
        public double LearningRate { get; set; }
        public DateTime Timestamp { get; set; }
        public double LossDifference => TrainLoss - ValLoss;
        public double AccuracyDifference => ValAccuracy - TrainAccuracy;
        public bool IsOverfitting => LossDifference > 0.1 && AccuracyDifference < -2.0;

        public override string ToString()
        {
            return $"Epoch {Epoch}: Train={TrainLoss:F4}({TrainAccuracy:F2}%), Val={ValLoss:F4}({ValAccuracy:F2}%), LR={LearningRate:E2}";
        }
    }
}
