using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;

namespace Курс.Core.Architecture
{
    public class ConvLayer : Layer
    {
        public int Filters { get; set; }
        public int KernelSize { get; set; }
        public int Stride { get; set; }
        public int Padding { get; set; }
        public bool UseBatchNorm { get; set; }

        public ConvLayer(string name, int filters, int kernelSize,
                        string activation = "relu", int stride = 1,
                        bool useBatchNorm = true)
            : base(name, "conv", activation)
        {
            Filters = filters;
            KernelSize = kernelSize;
            Stride = stride;
            Padding = 0;
            UseBatchNorm = useBatchNorm;
        }

        public override Module<Tensor, Tensor> CreateModule(int inputChannels = -1, long inputUnits = -1)
        {
            if (inputChannels <= 0)
                throw new ArgumentException("Для сверточного  inputChannels");

            var modules = new List<Module<Tensor, Tensor>>();

            var conv = Conv2d(inputChannels, Filters, KernelSize,
                                stride: Stride, padding: Padding);
            modules.Add(conv);

            if (UseBatchNorm)
            {
                var bn = BatchNorm2d(Filters);
                modules.Add(bn);
            }

            Module<Tensor, Tensor> activationModule = Activation switch
            {
                "relu" => ReLU(),
                "leaky_relu" => LeakyReLU(0.1),
                "sigmoid" => Sigmoid(),
                "tanh" => Tanh(),
                _ => ReLU()
            };
            modules.Add(activationModule);

            return Sequential(modules.ToArray());
        }

        public override Tensor ApplyActivation(Tensor x)
        {
            return Activation switch
            {
                "relu" => functional.relu(x),
                "leaky_relu" => functional.leaky_relu(x, 0.1),
                "sigmoid" => sigmoid(x),
                "tanh" => tanh(x),
                _ => functional.relu(x)
            };
        }

        public override Layer Clone()
        {
            return new ConvLayer(Name, Filters, KernelSize, Activation, Stride, UseBatchNorm);
        }

        public override bool Validate()
        {
            if (Filters <= 0 || Filters > 1024)
                return false;

            if (KernelSize < 1 || KernelSize > 7 || KernelSize % 2 == 0)
                return false;

            if (Stride < 1 || Stride > 3)
                return false;

            return true;
        }

        public override string GetDescription()
        {
            return $"[CONV] {Name}: {Filters} filters, {KernelSize}x{KernelSize}, " +
                   $"stride {Stride}, activation: {Activation}, BN: {UseBatchNorm}";
        }

        public override (int channels, int height, int width) CalculateOutputSize(
            int inputChannels, int inputHeight, int inputWidth)
        {
            int outputHeight = (inputHeight + 2 * Padding - KernelSize) / Stride + 1;
            int outputWidth = (inputWidth + 2 * Padding - KernelSize) / Stride + 1;

            return (Filters, outputHeight, outputWidth);
        }
    }
}
