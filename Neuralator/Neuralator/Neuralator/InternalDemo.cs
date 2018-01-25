using System;

namespace Neuralator
{
    public static class InternalDemo
    {
        public static void Run()
        {
            NeuralNet Winner = Neuralator.Neuralate();

            Console.WriteLine("Winner!");
            Console.WriteLine("Score out of 256: {0}", Winner.TotalScore);
            Console.WriteLine("Confidence: {0}", Winner.TotalAvgConfidence);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
