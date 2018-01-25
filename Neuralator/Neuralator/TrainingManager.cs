using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuralator
{
    internal static class TrainingManager
    {
        internal static List<TrainingDatum> TrainingSet;

        internal static void PopulateTrainingSet()
        {
            TrainingSet = new List<TrainingDatum>();

            // sin function
            for (int input = 0; input < Neuralator.NumberOfTrainingDatumToUse; input += 1)
            {
                List<float> InputValues = new List<float>();

                for (int iter = 0; iter < 8; iter += 1)
                {
                    InputValues.Add((input >> iter) & 0x1);
                }

                TrainingSet.Add(new TrainingDatum((float)Math.Sin(input * (2 * Math.PI / byte.MaxValue)), InputValues));
            }

            //using (StreamReader TrainingFile = new StreamReader(@"../../../Neuralator/DataSets/HandwrittenDigits/optdigits.tra"))
            //{
            //    while(TrainingFile.Peek() > 0)
            //    {
            //        string[] DatumTokens = TrainingFile.ReadLine().Split(',');
            //        List<float> DatumInputs = new List<float>();

            //        for (int iter = 0; iter < DatumTokens.Count() - 1; iter += 1)
            //        {
            //            DatumInputs.Add((float)(Int32.Parse(DatumTokens[iter]) / 16.0));
            //        }

            //        TrainingSet.Add(new TrainingDatum((float)Int32.Parse(DatumTokens.Last()), DatumInputs));
            //    }
            //}
        }
    }
}
