using ICSharpCode.SharpZipLib.Tar;
using SharpCompress.Archives;
using SharpCompress.Archives.Rar;
using SharpCompress.Archives.Zip;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NAS.Data
{
    public class CyrillicDataset
    {
        public string DataPath { get; private set; }
        public Dictionary<string, int> ClassToLabel { get; private set; }
        public Dictionary<int, string> LabelToClass { get; private set; }

        public List<(string imagePath, int label)> TrainSamples { get; set; }
        public List<(string imagePath, int label)> ValSamples { get; set; }

        public CyrillicDataset(string dataPath, double trainRatio = 0.8, int? seed = null)
        {
            DataPath = dataPath;
            ClassToLabel = new Dictionary<string, int>();
            LabelToClass = new Dictionary<int, string>();
            TrainSamples = new List<(string, int)>();
            ValSamples = new List<(string, int)>();

            LoadAndSplitDataset(trainRatio, seed);
        }

        private void LoadAndSplitDataset(double trainRatio, int? seed)
        {
            Console.WriteLine($"  Загрузка и разделение данных из: {DataPath}");
            Console.WriteLine($"   Соотношение: {trainRatio * 100}% train / {(1 - trainRatio) * 100}% validation");

            if (!Directory.Exists(DataPath))
            {
                Console.WriteLine($"  Папка с данными не найдена: {DataPath}");

                string zipFilePath = FindArchive();

                if (!string.IsNullOrEmpty(zipFilePath) && File.Exists(zipFilePath))
                {
                    Console.WriteLine($"  Найден ZIP архив: {zipFilePath}");
                    Console.WriteLine($"  Распаковываем архив в: {DataPath}");

                    ExtractArchive(zipFilePath, Path.GetDirectoryName(zipFilePath));

                    Console.WriteLine($"  Распаковка завершена");

                    // Проверяем снова после распаковки
                    if (!Directory.Exists(DataPath))
                    {
                        throw new DirectoryNotFoundException($"Папка с данными не найдена после распаковки: {DataPath}");
                    }
                }
                else
                {
                    throw new DirectoryNotFoundException($"Папка с данными не найдена: {DataPath}. ZIP архив также не найден.");
                }
            }

            var random = seed.HasValue ? new Random(seed.Value) : new Random();
            var allSamples = new List<(string imagePath, int label)>();

            var classDirectories = Directory.GetDirectories(DataPath)
                .Where(dir => !Path.GetFileName(dir).StartsWith("."))
                .OrderBy(dir => Path.GetFileName(dir))
                .ToArray();

            if (classDirectories.Length == 0)
                throw new InvalidDataException($"В папке {DataPath} не найдено подпапок с классами");

            int label = 0;
            foreach (var classDir in classDirectories)
            {
                string className = Path.GetFileName(classDir);
                ClassToLabel[className] = label;
                LabelToClass[label] = className;
                label++;
            }

            foreach (var classDir in classDirectories)
            {
                string className = Path.GetFileName(classDir);
                int classLabel = ClassToLabel[className];

                var imageFiles = Directory.GetFiles(classDir)
                    .Where(IsImageFile)
                    .Select(file => (file, classLabel))
                    //.Take(20)
                    .ToList();

                imageFiles = imageFiles.OrderBy(x => random.Next()).ToList();

                int trainCount = (int)(imageFiles.Count * trainRatio);

                TrainSamples.AddRange(imageFiles.Take(trainCount));
                ValSamples.AddRange(imageFiles.Skip(trainCount));

                Console.WriteLine($"  {className}: {imageFiles.Count} всего -> {trainCount} train, {imageFiles.Count - trainCount} val");
            }

            TrainSamples = TrainSamples.OrderBy(x => random.Next()).ToList();
            ValSamples = ValSamples.OrderBy(x => random.Next()).ToList();

            Console.WriteLine($"Разделение завершено:");
            Console.WriteLine($"   Train: {TrainSamples.Count} изображений");
            Console.WriteLine($"   Validation: {ValSamples.Count} изображений");
            Console.WriteLine($"   Всего классов: {ClassToLabel.Count}");
        }

        private string FindArchive()
        {
            try
            {
                string assemblyDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
                string[] zipFiles = Directory.GetFiles(assemblyDir, "*.zip");

                if (zipFiles.Length > 0)
                {
                    return zipFiles[0]; 
                }

              

                return null;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Ошибка при поиске ZIP архива: {ex.Message}");
                return null;
            }
        }

        private void ExtractArchive(string zipFilePath, string extractDirectory)
        {
            try
            {
                Directory.CreateDirectory(extractDirectory);

                using (var archive = ZipArchive.Open(zipFilePath))
                {
                    foreach (var entry in archive.Entries)
                    {
                        if (!entry.IsDirectory)
                        {

                            string entryOutputPath = Path.Combine(extractDirectory, entry.Key);

                            var directoryPath = Path.GetDirectoryName(entryOutputPath);
                            if (!string.IsNullOrEmpty(directoryPath))
                            {
                                Directory.CreateDirectory(directoryPath);
                            }

                            entry.WriteToFile(entryOutputPath);
                        }
                    }
                }

                Console.WriteLine($"  Архив успешно распакован из {zipFilePath} в {extractDirectory}");
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Ошибка при распаковке архива {zipFilePath}: {ex.Message}", ex);
            }
        }

        private bool IsImageFile(string filePath)
        {
            var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff" };
            return extensions.Contains(Path.GetExtension(filePath).ToLower());
        }

        public int NumClasses => ClassToLabel.Count;
        public int TrainSize => TrainSamples.Count;
        public int ValSize => ValSamples.Count;
    }
}
