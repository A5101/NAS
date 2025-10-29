using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace Курс.Core.Training
{
    public class PrecomputedBatch
    {
        public Tensor Images { get; set; }
        public Tensor Labels { get; set; }

        public PrecomputedBatch(Tensor images, Tensor labels)
        {
            Images = images;
            Labels = labels;
        }

        public void Dispose()
        {
            Images?.Dispose();
            Labels?.Dispose();
        }
    }

    public class PrecomputedBatches : IDisposable
    {
        public List<PrecomputedBatch> TrainBatches { get; set; }
        public List<PrecomputedBatch> ValBatches { get; set; }

        public PrecomputedBatches()
        {
            TrainBatches = new List<PrecomputedBatch>();
            ValBatches = new List<PrecomputedBatch>();
        }

        public void Dispose()
        {
            foreach (var batch in TrainBatches) batch.Dispose();
            foreach (var batch in ValBatches) batch.Dispose();
        }
    }
}
