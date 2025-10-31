using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;

namespace NAS.Core.Architecture
{
    public class FullyConnectedLayer : Layer
    {
        public int Units { get; set; }
        public double DropoutRate { get; private set; }
        public bool UseBatchNorm { get; private set; }

        public FullyConnectedLayer(string name, int units,
                                 string activation = "relu",
                                 double dropoutRate = 0.0,
                                 bool useBatchNorm = false)
            : base(name, "dense", activation)
        {
            Units = units;
            DropoutRate = dropoutRate;
            UseBatchNorm = useBatchNorm;
        }

        public override Module<Tensor, Tensor> CreateModule(int inputChannels = -1, long inputUnits = -1)
        {
            if (inputUnits <= 0)
                throw new ArgumentException("Для полносвязного  inputUnits");

            var modules = new List<Module<Tensor, Tensor>>();

            var linear = Linear(inputUnits, Units);
            modules.Add(linear);

            if (UseBatchNorm)
            {
                var bn = BatchNorm1d(Units);
                modules.Add(bn);
            }

            if (Activation != "none")
            {
                Module<Tensor, Tensor> activationModule = Activation switch
                {
                    "relu" => ReLU(),
                    "leaky_relu" => LeakyReLU(0.1),
                    "sigmoid" => Sigmoid(),
                    "tanh" => Tanh(),
                    _ => ReLU()
                };
                modules.Add(activationModule);
            }

            if (DropoutRate > 0.0)
            {
                var dropout = Dropout(DropoutRate);
                modules.Add(dropout);
            }

            return Sequential(modules.ToArray());
        }

        public override Tensor ApplyActivation(Tensor x)
        {
            var activated = Activation switch
            {
                "relu" => functional.relu(x),
                "leaky_relu" => functional.leaky_relu(x, 0.1),
                "sigmoid" => sigmoid(x),
                "tanh" => tanh(x),
                "none" => x,
                _ => functional.relu(x)
            };

            if (DropoutRate > 0.0)
            {
                activated = functional.dropout(activated, DropoutRate, training: true);
            }

            return activated;
        }

        public override Layer Clone()
        {
            return new FullyConnectedLayer(Name, Units, Activation, DropoutRate, UseBatchNorm);
        }

        public override bool Validate()
        {
            if (Units <= 0 || Units > 8192)
                return false;

            if (DropoutRate < 0.0 || DropoutRate >= 1.0)
                return false;

            return true;
        }

        public override string GetDescription()
        {
            string dropoutInfo = DropoutRate > 0 ? $", dropout: {DropoutRate:F2}" : "";
            string bnInfo = UseBatchNorm ? ", BN: true" : "";
            return $"[DENSE] {Name}: {Units} neurons, activation: {Activation}{dropoutInfo}{bnInfo}";
        }

        public override (int channels, int height, int width) CalculateOutputSize(
            int inputChannels, int inputHeight, int inputWidth)
        {
            return (Units, 1, 1);
        }
    }
}
