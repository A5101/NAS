using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Курс
{
    public class Architecture
    {
        public List<Layer> Layers { get; private set; }
        public string Name { get; set; }
        public double Accuracy { get; set; }
        public double TrainingTime { get; set; }

        public Architecture(string name = "")
        {
            Layers = new List<Layer>();
            Name = name;
            Accuracy = 0.0;
            TrainingTime = 0.0;
        }

        public void AddLayer(Layer layer)
        {
            if (layer.Validate())
            {
                Layers.Add(layer);
            }
            else
            {
                throw new ArgumentException($"Некорректные параметры слоя: {layer.GetDescription()}");
            }
        }

        public void InsertLayer(int index, Layer layer)
        {
            if (layer.Validate())
            {
                Layers.Insert(index, layer);
            }
            else
            {
                throw new ArgumentException($"Некорректные параметры слоя: {layer.GetDescription()}");
            }
        }

        public void RemoveLayer(int index)
        {
            if (index >= 0 && index < Layers.Count)
            {
                Layers.RemoveAt(index);
            }
        }

        public Architecture Clone()
        {
            var cloned = new Architecture(Name + "_clone");
            foreach (var layer in Layers)
            {
                cloned.AddLayer(layer.Clone());
            }
            cloned.Accuracy = Accuracy;
            cloned.TrainingTime = TrainingTime;
            return cloned;
        }

        public bool Validate()
        {
            if (Layers.Count == 0)
                return false;

            // Проверяем, что есть хотя бы один сверточный и выходной слой
            bool hasConv = Layers.Any(l => l.Type == "conv");
            bool hasOutput = Layers.Any(l => l.Type == "output");

            if (!hasConv || !hasOutput)
                return false;

            // Проверяем все слои
            foreach (var layer in Layers)
            {
                if (!layer.Validate())
                    return false;
            }

            return true;
        }

        public string GetSummary()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"АРХИТЕКТУРА: {Name}");
            sb.AppendLine($"Точность: {Accuracy:F2}%, Время обучения: {TrainingTime:F1}с");
            sb.AppendLine("=".PadRight(60, '='));

            int convCount = 0, poolCount = 0, denseCount = 0;

            for (int i = 0; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                sb.AppendLine($"{i + 1,2}. {layer.GetDescription()}");

                // Считаем типы слоев
                switch (layer.Type)
                {
                    case "conv": convCount++; break;
                    case "pool": poolCount++; break;
                    case "dense": denseCount++; break;
                }
            }

            sb.AppendLine("=".PadRight(60, '='));
            sb.AppendLine($"СТАТИСТИКА: {convCount} CONV, {poolCount} POOL, {denseCount} DENSE, {Layers.Count} всего");

            return sb.ToString();
        }

        // Вычисление общего размера модели
        public (int channels, int height, int width) CalculateFinalSize(int inputChannels = 1,
                                                                       int inputHeight = 64,
                                                                       int inputWidth = 64)
        {
            int channels = inputChannels;
            int height = inputHeight;
            int width = inputWidth;

            foreach (var layer in Layers)
            {
                if (layer.Type != "output") // Output layer не меняет spatial dimensions
                {
                    (channels, height, width) = layer.CalculateOutputSize(channels, height, width);
                }
            }

            return (channels, height, width);
        }

        // Проверка совместимости слоев
        public bool CheckLayerCompatibility(int channels = 3, int height = 64, int width = 64)
        {
            if (Layers.Count < 2) return true;

            foreach (var layer in Layers)
            {
                if (layer.Type == "flatten") continue;

                try
                {
                    var newSize = layer.CalculateOutputSize(channels, height, width);
                    channels = newSize.channels;
                    height = newSize.height;
                    width = newSize.width;

                    // Проверяем, что размеры не стали отрицательными
                    if (height <= 0 || width <= 0)
                        return false;
                }
                catch
                {
                    return false;
                }
            }

            return true;
        }
    }
}
