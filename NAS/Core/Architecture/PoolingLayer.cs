using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;

namespace NAS.Core.Architecture
{
    public class PoolingLayer : Layer
    {
        public string PoolType { get; set; } // "max", "avg"
        public int PoolSize { get; set; }
        public int Stride { get; set; }

        public PoolingLayer(string name, string poolType = "max",
                           int poolSize = 2, int stride = 2)
            : base(name, "pool", "none") 
        {
            PoolType = poolType;
            PoolSize = poolSize;
            Stride = stride;
        }

        public override Module<Tensor, Tensor> CreateModule(int inputChannels = -1, long inputUnits = -1)
        {
            Module<Tensor, Tensor> poolModule;

            if (PoolType == "max")
            {
                poolModule = MaxPool2d(new long[] { PoolSize, PoolSize },
                                         new long[] { Stride, Stride });
            }
            else
            {
                poolModule = AvgPool2d(new long[] { PoolSize, PoolSize },
                                         new long[] { Stride, Stride });
            }

            return poolModule;
        }

        public override Tensor ApplyActivation(Tensor x)
        {
            return x;
        }

        public override Layer Clone()
        {
            return new PoolingLayer(Name, PoolType, PoolSize, Stride);
        }

        public override bool Validate()
        {
            if (PoolSize < 2 || PoolSize > 4)
                return false;

            if (Stride < 1 || Stride > 4)
                return false;

            if (PoolType != "max" && PoolType != "avg")
                return false;

            return true;
        }

        public override string GetDescription()
        {
            return $"[POOL] {Name}: {PoolType} pooling, {PoolSize}x{PoolSize}, stride {Stride}";
        }

        public override (int channels, int height, int width) CalculateOutputSize(
        int inputChannels, int inputHeight, int inputWidth)
        {
            int outputHeight = (int)Math.Floor((double)(inputHeight - PoolSize) / Stride) + 1;
            int outputWidth = (int)Math.Floor((double)(inputWidth - PoolSize) / Stride) + 1;

            return (inputChannels, outputHeight, outputWidth);
        }
    }
}
