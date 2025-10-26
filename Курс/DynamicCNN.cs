using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace Курс
{
    public class DynamicCNN : Module<Tensor, Tensor>
    {
        private Sequential _layers;
        private Architecture _architecture;

        public DynamicCNN(Architecture architecture, int inputChannels = 1,
                         int inputHeight = 64, int inputWidth = 64, Device device = null)
            : base("DynamicCNN")
        {
            _architecture = architecture;
            device = device ?? (torch.cuda.is_available() ? CUDA : CPU);

            var modules = BuildModules(inputChannels, inputHeight, inputWidth);
            _layers = Sequential(modules.ToArray());

            RegisterComponents();

            if (device != null)
                this.to(device);

            Console.WriteLine($"Динамическая CNN создана: {modules.Count} модулей");
        }

        // Добавьте инициализацию весов в DynamicCNN
        private void InitializeWeights()
        {
            foreach (var module in this.modules())
            {
                switch (module)
                {
                    case Conv2d conv:
                        nn.init.kaiming_uniform_(conv.weight);
                        nn.init.constant_(conv.bias, 0);
                        break;
                    case Linear linear:
                        nn.init.xavier_uniform_(linear.weight);
                        nn.init.constant_(linear.bias, 0);
                        break;
                }
            }
        }

        private List<Module<Tensor, Tensor>> BuildModules(int inputChannels, int inputHeight, int inputWidth)
        {
            var modules = new List<Module<Tensor, Tensor>>();
            int currentChannels = inputChannels;
            int currentHeight = inputHeight;
            int currentWidth = inputWidth;
            long currentUnits = -1;
            bool flattenAdded = false;

            Console.WriteLine($"Построение модели из {_architecture.Layers.Count} слоев");
            Console.WriteLine($"Начальный размер: {currentChannels}x{currentHeight}x{currentWidth}");

            for (int i = 0; i < _architecture.Layers.Count; i++)
            {
                var layer = _architecture.Layers[i];

                try
                {
                    switch (layer)
                    {
                        case ConvLayer conv:
                            var convModule = conv.CreateModule(currentChannels);
                            modules.Add(convModule);
                            (currentChannels, currentHeight, currentWidth) =
                                conv.CalculateOutputSize(currentChannels, currentHeight, currentWidth);
                            Console.WriteLine($" {i + 1:00}. {conv.Name} -> {currentChannels}x{currentHeight}x{currentWidth}");
                            break;

                        case PoolingLayer pool:
                            var poolModule = pool.CreateModule();
                            modules.Add(poolModule);
                            (currentChannels, currentHeight, currentWidth) =
                                pool.CalculateOutputSize(currentChannels, currentHeight, currentWidth);
                            Console.WriteLine($" {i + 1:00}. {pool.Name} -> {currentChannels}x{currentHeight}x{currentWidth}");
                            break;

                        case CustomLayer custom when custom.Type == "flatten":
                            if (!flattenAdded)
                            {
                                modules.Add(new FlattenModule());
                                currentUnits = currentChannels * currentHeight * currentWidth;
                                flattenAdded = true;
                                Console.WriteLine($"{i + 1:00}. {custom.Name} -> {currentUnits} нейронов");
                            }
                            break;

                        case FullyConnectedLayer fc:
                            if (!flattenAdded)
                            {
                                throw new InvalidOperationException("Flatten слой должен быть перед полносвязными слоями");
                            }

                            var fcModule = fc.CreateModule(inputUnits: currentUnits);
                            currentUnits = fc.Units;
                            modules.Add(fcModule);
                            Console.WriteLine($" {i + 1:00}. {fc.Name} -> {fc.Units} нейронов");
                            break;

                        case OutputLayer output:
                            if (!flattenAdded)
                            {
                                throw new InvalidOperationException("Flatten слой должен быть перед выходным слоем");
                            }

                            var outputModule = output.CreateModule(inputUnits: currentUnits);
                            modules.Add(outputModule);
                            Console.WriteLine($" {i + 1:00}. {output.Name} -> {output.NumClasses} классов");
                            break;

                        default:
                            throw new NotSupportedException($"Неизвестный тип слоя: {layer.GetType().Name}");
                    }

                    // Проверяем, что размеры не стали некорректными
                    if (currentHeight <= 0 || currentWidth <= 0)
                    {
                        throw new InvalidOperationException(
                            $"Некорректный размер после слоя {i + 1}: {currentHeight}x{currentWidth}");
                    }
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException(
                        $"Ошибка построения слоя {i + 1} ({layer.GetDescription()}): {ex.Message}", ex);
                }
            }

            return modules;
        }

        public override Tensor forward(Tensor x)
        {
            return _layers.forward(x);
        }

        // Метод для получения информации о модели
        public string GetModelInfo()
        {
            var parameters = this.parameters().Count();
            var totalParams = this.parameters().Sum(p => p.numel());

            return $"Модель: {_architecture.Name}\n" +
                   $"Слоев: {_architecture.Layers.Count}\n" +
                   $"Параметров: {totalParams:N0}\n" +
                   $"Точность: {_architecture.Accuracy:F2}%";
        }

        // Метод для тестирования прямого прохода
        public void TestForwardPass(int batchSize = 1, int inputChannels = 1,
                            int inputHeight = 64, int inputWidth = 64)
        {
            try
            {
                Console.WriteLine($"\nТЕСТ ПРЯМОГО ПРОХОДА:");
                Console.WriteLine($"   Батч: {batchSize}x{inputChannels}x{inputHeight}x{inputWidth}");
                Console.WriteLine($"   Устройство модели: {this.parameters().First().device}");

                // СОЗДАЕМ ТЕНЗОР НА ТОМ ЖЕ УСТРОЙСТВЕ, ЧТО И МОДЕЛЬ
                var modelDevice = this.parameters().First().device;
                var input = torch.randn(new long[] { batchSize, inputChannels, inputHeight, inputWidth }).to(modelDevice);

                Console.WriteLine($"   Устройство input: {input.device}");

                var output = this.forward(input);

                Console.WriteLine($"Вход:  {string.Join("x", input.shape)}");
                Console.WriteLine($"Выход: {string.Join("x", output.shape)}");
                Console.WriteLine($" Min: {output.min().item<float>():F4}, Max: {output.max().item<float>():F4}");

                // Очищаем память
                input.Dispose();
                output.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   Ошибка: {ex.Message}");
                Console.WriteLine($"   StackTrace: {ex.StackTrace}");
            }
        }
    }
}