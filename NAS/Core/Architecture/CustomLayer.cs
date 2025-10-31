using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;

namespace NAS.Core.Architecture
{
    public class CustomLayer : Layer
    {
        public CustomLayer(string name, string type, string activation = "none")
            : base(name, type, activation)
        {
        }

        public override Module<Tensor, Tensor> CreateModule(int inputChannels = -1, long inputUnits = -1)
        {
            return new FlattenModule();
        }

        public override Tensor ApplyActivation(Tensor x)
        {
            return x;
        }

        public override Layer Clone()
        {
            return new CustomLayer(Name, Type, Activation);
        }

        public override bool Validate()
        {
            return true;
        }

        public override string GetDescription()
        {
            return $"[{Type.ToUpper()}] {Name}";
        }

        public override (int channels, int height, int width) CalculateOutputSize(
            int inputChannels, int inputHeight, int inputWidth)
        {
            if (Type == "flatten")
            {
                return (inputChannels * inputHeight * inputWidth, 1, 1);
            }

            return (inputChannels, inputHeight, inputWidth);
        }
    }

    public class FlattenModule : Module<Tensor, Tensor>
    {
        public FlattenModule() : base("Flatten")
        {
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return x.flatten(1); 
        }
    }
}
