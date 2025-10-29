using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Курс.NAS.Models
{
    public class GeneticConfig
    {
        public int PopulationSize { get; set; } = 20;
        public int Generations { get; set; } = 50;
        public double CrossoverRate { get; set; } = 0.8;
        public double MutationRate { get; set; } = 0.3;
        public double EliteRatio { get; set; } = 0.2;
        public int TournamentSize { get; set; } = 3;
        public int MinLayers { get; set; } = 4;
        public int MaxLayers { get; set; } = 15;
        public int EpochsPerEvaluation { get; set; } = 10;
    }
}
