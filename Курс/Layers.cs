using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp;

namespace Курс
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

        // Абстрактные методы для создания PyTorch слоев
        public abstract Module<Tensor, Tensor> CreateModule(int inputChannels = -1, long inputUnits = -1);
        public abstract Tensor ApplyActivation(Tensor x);
        public abstract Layer Clone();

        // Валидация параметров слоя
        public abstract bool Validate();

        // Информация о слое
        public virtual string GetDescription()
        {
            return $"{Type} layer '{Name}'";
        }

        // Вычисление выходного размера (для совместимости слоев)
        public abstract (int channels, int height, int width) CalculateOutputSize(
            int inputChannels, int inputHeight, int inputWidth);
    }

    public class ConvLayer : Layer
    {
        public int Filters { get; private set; }
        public int KernelSize { get; private set; }
        public int Stride { get; private set; }
        public int Padding { get; private set; }
        public bool UseBatchNorm { get; private set; }

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

            // 1. Conv2d слой
            var conv = nn.Conv2d(inputChannels, Filters, KernelSize,
                                stride: Stride, padding: Padding);
            modules.Add(conv);

            // 2. BatchNorm (если включен)
            if (UseBatchNorm)
            {
                var bn = nn.BatchNorm2d(Filters);
                modules.Add(bn);
            }

            // 3. Активация
            Module<Tensor, Tensor> activationModule = Activation switch
            {
                "relu" => nn.ReLU(),
                "leaky_relu" => nn.LeakyReLU(0.1),
                "sigmoid" => nn.Sigmoid(),
                "tanh" => nn.Tanh(),
                _ => nn.ReLU()
            };
            modules.Add(activationModule);

            // Создаем Sequential из всех модулей
            return nn.Sequential(modules.ToArray());
        }

        public override Tensor ApplyActivation(Tensor x)
        {
            return Activation switch
            {
                "relu" => torch.nn.functional.relu(x),
                "leaky_relu" => torch.nn.functional.leaky_relu(x, 0.1),
                "sigmoid" => torch.sigmoid(x),
                "tanh" => torch.tanh(x),
                _ => torch.nn.functional.relu(x) // По умолчанию ReLU
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

            if (KernelSize < 1 || KernelSize > 7 || KernelSize % 2 == 0) // Только нечетные размеры
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

    public class PoolingLayer : Layer
    {
        public string PoolType { get; private set; } // "max", "avg"
        public int PoolSize { get; private set; }
        public int Stride { get; private set; }

        public PoolingLayer(string name, string poolType = "max",
                           int poolSize = 2, int stride = 2)
            : base(name, "pool", "none") // Pooling не имеет активации
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
                poolModule = nn.MaxPool2d(new long[] { PoolSize, PoolSize },
                                         new long[] { Stride, Stride });
            }
            else
            {
                poolModule = nn.AvgPool2d(new long[] { PoolSize, PoolSize },
                                         new long[] { Stride, Stride });
            }

            return poolModule;
        }

        public override Tensor ApplyActivation(Tensor x)
        {
            return x; // Pooling не применяет активацию
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

    public class FullyConnectedLayer : Layer
    {
        public int Units { get; private set; }
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

            // 1. Linear слой
            var linear = nn.Linear(inputUnits, Units);
            modules.Add(linear);

            // 2. BatchNorm (если включен)
            if (UseBatchNorm)
            {
                var bn = nn.BatchNorm1d(Units);
                modules.Add(bn);
            }

            // 3. Активация (если не "none")
            if (Activation != "none")
            {
                Module<Tensor, Tensor> activationModule = Activation switch
                {
                    "relu" => nn.ReLU(),
                    "leaky_relu" => nn.LeakyReLU(0.1),
                    "sigmoid" => nn.Sigmoid(),
                    "tanh" => nn.Tanh(),
                    _ => nn.ReLU()
                };
                modules.Add(activationModule);
            }

            // 4. Dropout (если включен)
            if (DropoutRate > 0.0)
            {
                var dropout = nn.Dropout(DropoutRate);
                modules.Add(dropout);
            }

            return nn.Sequential(modules.ToArray());
        }

        public override Tensor ApplyActivation(Tensor x)
        {
            var activated = Activation switch
            {
                "relu" => torch.nn.functional.relu(x),
                "leaky_relu" => torch.nn.functional.leaky_relu(x, 0.1),
                "sigmoid" => torch.sigmoid(x),
                "tanh" => torch.tanh(x),
                "none" => x,
                _ => torch.nn.functional.relu(x)
            };

            // Применяем dropout если нужно
            if (DropoutRate > 0.0)
            {
                activated = torch.nn.functional.dropout(activated, DropoutRate, training: true);
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
            // Полносвязный слой преобразует в 1D вектор
            return (Units, 1, 1);
        }
    }

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
            return nn.Linear(inputUnits, NumClasses);
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
                // Flatten преобразует в 1D вектор
                return (inputChannels * inputHeight * inputWidth, 1, 1);
            }

            return (inputChannels, inputHeight, inputWidth);
        }
    }

    // Реализация Flatten модуля для TorchSharp
    public class FlattenModule : Module<Tensor, Tensor>
    {
        public FlattenModule() : base("Flatten")
        {
            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            return x.flatten(1); // flatten начиная с dimension 1 (после batch)
        }
    }
}
