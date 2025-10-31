using NAS.Core.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NAS.Core.Architecture
{
    /// <summary>
    /// Представляет конкретную архитектуру нейронной сети, содержащую последовательность слоев
    /// </summary>
    public class ConcreteArchitecture
    {
        /// <summary>
        /// Список слоев архитектуры
        /// </summary>
        public List<Layer> Layers { get; set; }

    

        /// <summary>
        /// Название архитектуры
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Точность архитектуры в процентах
        /// </summary>
        public double Accuracy { get; set; }

        /// <summary>
        /// Время обучения архитектуры в секундах
        /// </summary>
        public double TrainingTime { get; set; }

        /// <summary>
        /// Инициализирует новый экземпляр класса ConcreteArchitecture
        /// </summary>
        /// <param name="name">Название архитектуры (по умолчанию пустая строка)</param>
        public ConcreteArchitecture(string name = "")
        {
            Layers = new List<Layer>();
            Name = name;
            Accuracy = 0.0;
            TrainingTime = 0.0;
        }

        /// <summary>
        /// Добавляет слой в конец архитектуры после проверки его валидности
        /// </summary>
        /// <param name="layer">Слой для добавления</param>
        /// <exception cref="ArgumentException">Вызывается при некорректных параметрах слоя</exception>
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

        /// <summary>
        /// Вставляет слой в указанную позицию архитектуры после проверки его валидности
        /// </summary>
        /// <param name="index">Индекс позиции для вставки</param>
        /// <param name="layer">Слой для вставки</param>
        /// <exception cref="ArgumentException">Вызывается при некорректных параметрах слоя</exception>
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

        /// <summary>
        /// Удаляет слой из архитектуры по указанному индексу
        /// </summary>
        /// <param name="index">Индекс слоя для удаления</param>
        public void RemoveLayer(int index)
        {
            if (index >= 0 && index < Layers.Count)
            {
                Layers.RemoveAt(index);
            }
        }

        /// <summary>
        /// Создает глубокую копию текущей архитектуры
        /// </summary>
        /// <returns>Новый экземпляр ConcreteArchitecture с копией всех слоев</returns>
        public ConcreteArchitecture Clone()
        {
            var cloned = new ConcreteArchitecture(Name + "_clone");
            foreach (var layer in Layers)
            {
                cloned.AddLayer(layer.Clone());
            }
            cloned.Accuracy = Accuracy;
            cloned.TrainingTime = TrainingTime;
            return cloned;
        }

        /// <summary>
        /// Проверяет валидность всей архитектуры
        /// </summary>
        /// <returns>True если архитектура валидна, иначе False</returns>
        public bool Validate()
        {
            if (Layers.Count == 0)
                return false;

            bool hasConv = Layers.Any(l => l.Type == "conv");
            bool hasOutput = Layers.Any(l => l.Type == "output");

            if (!hasConv || !hasOutput)
                return false;

            foreach (var layer in Layers)
            {
                if (!layer.Validate())
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Формирует текстовое описание архитектуры со статистикой
        /// </summary>
        /// <returns>Строка с детальным описанием архитектуры</returns>
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

        /// <summary>
        /// Вычисляет размер выходного тензора после прохождения через все слои архитектуры
        /// </summary>
        /// <param name="inputChannels">Количество входных каналов (по умолчанию 1)</param>
        /// <param name="inputHeight">Высота входного изображения (по умолчанию 64)</param>
        /// <param name="inputWidth">Ширина входного изображения (по умолчанию 64)</param>
        /// <returns>Кортеж (channels, height, width) с размерами выходного тензора</returns>
        public (int channels, int height, int width) CalculateFinalSize(int inputChannels = 1,
                                                                       int inputHeight = 64,
                                                                       int inputWidth = 64)
        {
            int channels = inputChannels;
            int height = inputHeight;
            int width = inputWidth;

            foreach (var layer in Layers)
            {
                if (layer.Type != "output")
                {
                    (channels, height, width) = layer.CalculateOutputSize(channels, height, width);
                }
            }

            return (channels, height, width);
        }

        /// <summary>
        /// Проверяет совместимость слоев архитектуры по размерам данных
        /// </summary>
        /// <param name="channels">Количество входных каналов (по умолчанию 3)</param>
        /// <param name="height">Высота входного изображения (по умолчанию 64)</param>
        /// <param name="width">Ширина входного изображения (по умолчанию 64)</param>
        /// <returns>True если все слои совместимы, иначе False</returns>
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