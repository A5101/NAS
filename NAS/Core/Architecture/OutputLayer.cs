using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;

namespace Курс.Core.Architecture
{
    public class OutputLayer : Layer
    {
        public int NumClasses { get; private set; }

        public OutputLayer(string name, int numClasses)
            : base(name, "output", "none") // Output layer обычно без активации (применяется в loss)
        {
            NumClasses = numClasses;
        }

        public override Module<Tensor, Tensor> CreateModule(int inputChannels = -1, long inputUnits = -1)
        {
            if (inputUnits <= 0)
                throw new ArgumentException("Для выходного слоя必须指定 inputUnits");

            // Output layer - просто Linear слой без активации
            // (активация softmax обычно применяется в loss функции)
            return Linear(inputUnits, NumClasses);
        }

        public override Tensor ApplyActivation(Tensor x)
        {
            return x; // Без активации - будет использоваться с CrossEntropyLoss
        }

        public override Layer Clone()
        {
            return new OutputLayer(Name, NumClasses);
        }

        public override bool Validate()
        {
            return NumClasses > 0;
        }

        public override string GetDescription()
        {
            return $"[OUTPUT] {Name}: {NumClasses} classes (softmax)";
        }

        public override (int channels, int height, int width) CalculateOutputSize(
            int inputChannels, int inputHeight, int inputWidth)
        {
            return (NumClasses, 1, 1);
        }
    }
}
