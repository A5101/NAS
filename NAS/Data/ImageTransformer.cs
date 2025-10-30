using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace Курс.Data
{
    public class ImageTransformer
    {
        private readonly int _targetWidth;
        private readonly int _targetHeight;
        private readonly bool _augmentation;

        public ImageTransformer(int width = 64, int height = 64, bool augmentation = false)
        {
            _targetWidth = width;
            _targetHeight = height;
            _augmentation = augmentation;
        }

        public Tensor Transform(string imagePath, Device device = null)
        {
            device = device ?? CPU;

            try
            {
                using (var image = Image.Load<Rgba32>(imagePath))
                {
                    image.Mutate(x => x.Resize(_targetWidth, _targetHeight));

                    var tensor = zeros(new long[] { 1, _targetHeight, _targetWidth });

                    int nonZeroPixels = 0;
                    int transparentPixels = 0;

                    for (int y = 0; y < _targetHeight; y++)
                    {
                        for (int x = 0; x < _targetWidth; x++)
                        {
                            var pixel = image[x, y];

                            if (pixel.A == 0)
                            {
                                tensor[0, y, x] = 1.0f;
                                transparentPixels++;
                            }
                            else
                            {
                                var grayValue = (0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B) / 255.0f;
                                tensor[0, y, x] = grayValue;
                                nonZeroPixels++;
                            }
                        }
                    }

                    return tensor.to(device);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка загрузки {imagePath}: {ex.Message}");
                return zeros(new long[] { 1, _targetHeight, _targetWidth }).to(device);
            }
        }

        private void ApplyAugmentation(IImageProcessingContext context)
        {
            var random = new Random();

            if (random.NextDouble() > 0.5)
            {
                var angle = random.NextDouble() * 20 - 10; 
                context.Rotate((float)angle);
            }

            if (random.NextDouble() > 0.5)
            {
                var brightness = (float)(random.NextDouble() * 0.4 - 0.2); 
                context.Brightness((float)(1.0 + brightness));
            }
        }

        private Tensor ImageToTensor(Image<L8> image)
        {
            var width = image.Width;
            var height = image.Height;

            var tensor = zeros(new long[] { 1, height, width });

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var pixel = image[x, y];
                    tensor[0, y, x] = pixel.PackedValue / 255.0f; 
                }
            }

            return tensor;
        }
        public Tensor Normalize(Tensor tensor)
        {
            var mean = torch.tensor(new float[] { 0.5f }).reshape(1, 1, 1).to(tensor.device);
            var std = torch.tensor(new float[] { 0.5f }).reshape(1, 1, 1).to(tensor.device);

            return (tensor - mean) / std;
        }
    }
}