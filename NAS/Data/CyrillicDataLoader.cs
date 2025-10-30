using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using SixLabors.ImageSharp;
using Курс.Core.Training;

namespace Курс.Data
{
    public class CyrillicDataLoader
    {
        private readonly CyrillicDataset _dataset;
        private readonly ImageTransformer _trainTransformer;
        private readonly ImageTransformer _valTransformer;
        private readonly Random _random;

        public CyrillicDataLoader(string dataPath, int imageSize = 64, int? seed = null)
        {
            _dataset = new CyrillicDataset(dataPath, trainRatio: 0.8, seed: seed);
            _trainTransformer = new ImageTransformer(imageSize, imageSize, augmentation: true);
            _valTransformer = new ImageTransformer(imageSize, imageSize, augmentation: false);
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
        }

        public PrecomputedBatches PrecomputeBatches(int batchSize, Device device = null)
        {
            device = device ?? CPU;
            var batches = new PrecomputedBatches();

            Console.WriteLine($"ПРЕДВАРИТЕЛЬНОЕ СОЗДАНИЕ БАТЧЕЙ...");
            Console.WriteLine($"   Batch size: {batchSize}");
            Console.WriteLine($"   Устройство: {device}");

            Console.WriteLine($"   Создание train батчей...");
            batches.TrainBatches = CreateBatches(_dataset.TrainSamples, _trainTransformer, batchSize, device, "Train");

            Console.WriteLine($"   Создание val батчей...");
            batches.ValBatches = CreateBatches(_dataset.ValSamples, _valTransformer, batchSize, device, "Val");

            Console.WriteLine($" Создано: {batches.TrainBatches.Count} train батчей, {batches.ValBatches.Count} val батчей");

            return batches;
        }

        private List<PrecomputedBatch> CreateBatches(List<(string path, int label)> samples,
                                                   ImageTransformer transformer,
                                                   int batchSize, Device device, string name)
        {
            var batches = new List<PrecomputedBatch>();
            var shuffledSamples = samples.OrderBy(x => _random.Next()).ToList();

            for (int i = 0; i < shuffledSamples.Count; i += batchSize)
            {
                var batchSamples = shuffledSamples.Skip(i).Take(batchSize).ToList();

                var images = new List<Tensor>();
                var labels = new List<Tensor>();

                foreach (var sample in batchSamples)
                {
                    var image = transformer.Transform(sample.path, device);
                    image = transformer.Normalize(image);
                    var label = tensor(sample.label, int64).to(device);

                    images.Add(image);
                    labels.Add(label);
                }

                var batchImages = stack(images.ToArray());
                var batchLabels = stack(labels.ToArray());

                batches.Add(new PrecomputedBatch(batchImages, batchLabels));

                foreach (var img in images) img.Dispose();
                foreach (var lbl in labels) lbl.Dispose();

                if (batches.Count % 10 == 0)
                {
                    Console.WriteLine($"     {name}: создано {batches.Count} батчей");
                }
            }

            return batches;
        }

        public CyrillicDataset Dataset => _dataset;

        public void PrintDatasetInfo()
        {
            Console.WriteLine($"\nИНФОРМАЦИЯ О ДАТАСЕТЕ:");
            Console.WriteLine($"   Train samples: {_dataset.TrainSize}");
            Console.WriteLine($"   Validation samples: {_dataset.ValSize}");
            Console.WriteLine($"   Classes: {_dataset.NumClasses}");
        }
    }
}
