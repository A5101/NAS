using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;

namespace Курс.Core.Architecture
{
    public abstract class Layer
    {
        public string Name { get; protected set; }
        public string Type { get; protected set; }
        public string Activation { get; protected set; }

        protected Layer(string name, string type, string activation = "relu")
        {
            Name = name;
            Type = type;
            Activation = activation;
        }

        public abstract Module<Tensor, Tensor> CreateModule(int inputChannels = -1, long inputUnits = -1);
        public abstract Tensor ApplyActivation(Tensor x);
        public abstract Layer Clone();

        public abstract bool Validate();

        public virtual string GetDescription()
        {
            return $"{Type} layer '{Name}'";
        }

        public abstract (int channels, int height, int width) CalculateOutputSize(
            int inputChannels, int inputHeight, int inputWidth);
    }
}
