using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;

namespace Neuralator
{
    /// <summary>
    /// Entry point into the system. Contains a set of global configuration parameters, as well
    /// as the functions that allow a user to train the network to a supplied training function.
    /// </summary>
    internal static class Neuralator
    {
        /// <summary>
        /// This provides the configuration both for the number of NeuralLayers in each NeuralNet,
        /// but also the number of Neurons in each layer. Each index in the list represents the 
        /// number of Neurons in that layer, with the first entry being the number if input neurons
        /// and the last entry being the number of output Neurons.
        /// For example: { 256, 64, 64, 10 } describes a 4 layer neural network that has 256 input
        /// Neurons in the first NeuralLayer, and 10 output Neurons in the last NeuralLayer.
        /// </summary>
        internal readonly static List<UInt16> NeuralLayerInfo = new List<UInt16> { NeuralLayerCountFirst, 16, NeuralLayerCountLast };
        internal const UInt16 NeuralLayerCountFirst = 8;
        internal const UInt16 NeuralLayerCountLast  = 21;

        /// <summary>
        /// Number of generations to run for.
        /// </summary>
        internal const int NumberOfGenerations = 2048;

        /// <summary>
        /// Number of NeuralNet members in each generation.
        /// </summary>
        internal const int NetsPerGeneration = 2 * SurvivorsPerGeneration * SurvivorsPerGeneration;

        /// <summary>
        /// Number of NeuralNets that will survive each generation. The survivors will breed,
        /// then them and their offspring will constitute the next generation.
        /// </summary>
        internal const int SurvivorsPerGeneration = 16;

        /// <summary>
        /// Method for combining two NeuralNetworks to create a combined child NeuralNet for
        /// the next generation.
        /// </summary>
        internal enum EnumBreedingType { AlwaysRandom, Human, WeightedPull, AverageValue }
        internal static EnumBreedingType BreedingType = EnumBreedingType.Human;

        /// <summary>
        /// During breeding of two NeuralNets to create a child NeuralNet, each connection
        /// has a MutationRate chance of being assigned a random value. 0.0 to 1.0.
        /// </summary>
        internal const float BaseMutationRate = 0.01f;

        /// <summary>
        /// For each generation that does not provide an improvement in survival odds, increase
        /// the mutation rate used during breeding to provide more varied offspring.
        /// </summary>
        internal const float MutationRateIncreasePerFailedGeneration = 0.0005f;

        /// <summary>
        /// Highest possible mutation rate.
        /// </summary>
        internal const float MaxMutationRate = 0.01f;

        /// <summary>
        /// A Neuron connects to all Neurons in the next NeuralLayer. However, some of these
        /// connections may be 'dead' (they will not propogate a signal, achieved by setting the
        /// Dendrite.ConnectionStrength to 0). This value is the percent of all Dendrites that 
        /// will be valid connections. (1 - DendriteFillRate == Percent Of Dead Connections).
        /// 0.0 to 1.0.
        /// </summary>
        internal const float DendriteFillRate = 0.4f;

        /// <summary>
        /// Minimum value that will be propgated through a Dendrite connection.
        /// The value that is propgated is InputNeuron.NeuronValue * Dendrite.ConnectionStrength.
        /// If the result is lower than MinimumActionPotential, the output Neuron will not be
        /// signaled.
        /// </summary>
        internal const float MinimumActionPotential = 0.0f;

        /// <summary>
        /// Maximum possible connection strength a Dendrite can be randomly assigned. 0.0 to 1.0
        /// Values greater than 1.0 will amplify the signal, which may be desirable in some cases?
        /// </summary>
        internal const float MaxConnectionStrength = 1.0f;
        /// <summary>
        /// Minimum possible connection strength a Dendrite can be randomly assigned. 0.0 to 1.0.
        /// Note: The true minimum is always 0 for a dead connection, which is based on DendriteFillRate.
        /// </summary>
        internal const float MinConnectionStrength = 0.0f;

        /// <summary>
        /// Number of entries in the training set to use.
        /// </summary>
        internal const uint NumberOfTrainingDatumToUse = 256;

        /// <summary>
        /// When culling, all NeuralNets that produce the same error are automatically considered identical.
        /// </summary>
        internal const bool CullWithMatchingError = true;

        /// <summary>
        /// Variables for converting NeuralNet output value to be directly comparable to the output of TrainingFunction.
        /// </summary>
        internal const double TrainingFunctionMin = -1;
        internal const double TrainingFunctionMax = 1;
        internal const double InputMinimum = 0;
        internal const double InputMaximum = NeuralLayerCountLast - 1;
        internal const double Slope = (TrainingFunctionMax - TrainingFunctionMin) / (InputMaximum - InputMinimum);
        internal const double Resolution = Slope / 2;
        internal const double YIntercept = TrainingFunctionMin - (Slope * InputMinimum);

        /// <summary>
        /// Seed for the PRNG that drives all evolution.
        /// </summary>
        //internal const int RNGSeed = 5465422;

        internal enum ErrorSelectorType { SumOfSquares, SumOfRoots, SumOfCustom }
        private static ErrorSelectorType ErrorSelector = ErrorSelectorType.SumOfSquares;

        internal static bool SortByScore { get; private set; } = false;

        private static List<NeuralNet> GenerationMembers = new List<NeuralNet>();
        internal static readonly Random GlobalRandom = new Random();
        private static Stopwatch TotalTimer = new Stopwatch();
        private static Stopwatch PortionTimer = new Stopwatch();
        private static int GensSinceLastImprv = 0;
        private static int LastGenTopScore = 0;
        private static double LastGenLeastError = int.MaxValue;
        internal static object _lock = new object();

        internal static NeuralNet Neuralate()
        {
            TrainingManager.PopulateTrainingSet();

            for (int iter = 0; iter < NetsPerGeneration; iter += 1)
            {
                GenerationMembers.Add(new NeuralNet(NeuralLayerInfo));
            }

            // run all generations
            TotalTimer.Start();
            PortionTimer.Start();
            for (int Generation = 0; Generation < NumberOfGenerations; Generation += 1)
            {
                PortionTimer.Restart();
                // Run all members of the generation over the training set and score them
                Parallel.ForEach(GenerationMembers, Member =>
                {
                    Member.ClearScore();

                    // Loop over all inputs the current Member and score the total
                    for (int input = 0; input < NumberOfTrainingDatumToUse; input += 1)
                    {
                        // fill each net with input data for point
                        Member.ClearNetworkState();
                        Member.SetInputsToFirstLayer(TrainingManager.TrainingSet[input]);

                        Member.Propogate();
                        Member.GetConsensus();

                        var act = Actual(Member.OutputValue);
                        var exp = TrainingManager.TrainingSet[input].OutputValue;
                        var Error = Math.Abs(act - exp);

                        if (Error < Resolution)
                        {
                            Member.TotalScore += 1;
                        }
                        else
                        {
                            Member.SumOfSquaresError += Math.Pow(Error, 2);
                            Member.SumOfRootsError   += Math.Pow(Error, 0.5);
                            Member.SumOfCustomError  += (Error + 10 * Math.Sqrt(Error));
                        }

                        Member.TotalAvgConfidence += Member.Confidence;
                    }

                    //Member.TotalAvgConfidence = Math.Max(Member.TotalAvgConfidence, 0.01f);

                    Member.TotalAvgConfidence /= NumberOfTrainingDatumToUse;

                    //Member.SumOfSquaresError /= Member.TotalAvgConfidence;
                    //Member.SumOfRootsError   /= Member.TotalAvgConfidence;
                    //Member.SumOfCustomError  /= Member.TotalAvgConfidence;

                    switch (ErrorSelector)
                    {
                        case ErrorSelectorType.SumOfSquares:
                            Member.TotalError = Member.SumOfSquaresError;
                            break;

                        case ErrorSelectorType.SumOfRoots:
                            Member.TotalError = Member.SumOfRootsError;
                            break;

                        case ErrorSelectorType.SumOfCustom:
                            Member.TotalError = Member.SumOfCustomError;
                            break;

                        default:
                            Member.TotalError = Member.SumOfSquaresError;
                            break;
                    }
                });

                var PropogationTime = PortionTimer.Elapsed;
                PortionTimer.Restart();

                GenerationMembers.Sort();

                if (SortByScore ? (LastGenTopScore >= GenerationMembers.First().TotalScore) : (LastGenLeastError <= GenerationMembers.First().TotalError))
                {
                    GensSinceLastImprv += 1;

                    //if (SortByScore)
                    //{
                    //    if (GensSinceLastImprv >= 200)
                    //    {
                    //        return GenerationMembers[0];
                    //    }
                    //}
                    //if (ErrorSelector == ErrorSelectorType.SumOfSquares)
                    //{
                    //    if(GensSinceLastImprv == 100)
                    //    {
                    //        ErrorSelector = ErrorSelectorType.SumOfRoots;
                    //        LastGenTopScore = GenerationMembers.First().TotalScore;
                    //        LastGenLeastError = GenerationMembers.First().TotalError;
                    //    }
                    //}
                    //else
                    //{
                    //    if (GensSinceLastImprv >= 200)
                    //    {
                    //        SortByScore = true;
                    //    }
                    //}
                }
                else
                {
                    GensSinceLastImprv = 0;
                    LastGenTopScore = GenerationMembers.First().TotalScore;
                    LastGenLeastError = GenerationMembers.First().TotalError;
                }

                NeuralNet BestInGeneration = GenerationMembers[0];

                if ((GensSinceLastImprv == 20) || ((GensSinceLastImprv + 1) % 100 == 0))
                {
                    Cull();
                }

                BreedNextGeneration();

                DisplayGenerationStats(Generation, BestInGeneration, PropogationTime, PortionTimer.Elapsed);
            }
            TotalTimer.Stop();

            NeuralNet Winner = GenerationMembers[0];

            foreach (var Member in GenerationMembers)
            {
                if (Member.TotalScore > Winner.TotalScore)
                {
                    Winner = Member;
                }
            }

            return Winner;
        }


        internal static void Cull()
        {
            for (int iter = 0; iter < GenerationMembers.Count() - 1; iter += 1)
            {
                //if (GenerationMembers[0].TotalError == GenerationMembers[iter].TotalError)
                //{
                uint ProximalMembers = 0;

                for (int iter2 = 0; iter2 < GenerationMembers.Count(); iter2 += 1)
                {
                    if (iter == iter2)
                    {
                        continue;
                    }

                    var DistanceTo = GenerationMembers[iter].DistanceTo(GenerationMembers[iter2]);

                    if (DistanceTo <= 1)
                    {
                        if ((ProximalMembers == 10) || DistanceTo <= 0.1 || (GenerationMembers[iter].TotalError == GenerationMembers[iter2].TotalError))
                        {
                            GenerationMembers[iter2] = Breed(GenerationMembers[GlobalRandom.Next(SurvivorsPerGeneration + 1, GenerationMembers.Count())],
                                                             GenerationMembers[GlobalRandom.Next(SurvivorsPerGeneration + 1, GenerationMembers.Count())]);
                        }
                        else
                        {
                            ProximalMembers += 1;
                        }
                    }
                }
                //}
                //else
                //{
                //    GenerationMembers[iter] = Breed(GenerationMembers[GlobalRandom.Next(SurvivorsPerGeneration + 1, GenerationMembers.Count())],
                //                                    GenerationMembers[GlobalRandom.Next(SurvivorsPerGeneration + 1, GenerationMembers.Count())]);
                //}
            }
        }


        /// <summary>
        /// Interprets the output of the last NeuralLayer (the output layer) as a result that can be directly
        /// compared to the output of the training funcion.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal static double Actual(int x)
        {
            return ((Slope * x) + YIntercept);
        }


        internal static void BreedNextGeneration()
        {
            List<NeuralNet> NextGeneration = new List<NeuralNet>();

            // kill bottom n based on murder rate
            // breed winners until number of nets at netspergen
            for (int iter = 0; iter < 1; iter += 1)
            {
                NextGeneration.Add(GenerationMembers[iter]);
            }

            for (int iter = 0; iter < (int)Math.Sqrt(NetsPerGeneration); iter += 1)
            {
                //for (int iter2 = 0; iter2 < (int)Math.Sqrt(NetsPerGeneration); iter2 += 1)
                //{
                //    if (NextGeneration.Count() == GenerationMembers.Count())
                //    {
                //        break;
                //    }

                //    if (GenerationMembers[iter].DistanceTo(GenerationMembers[iter2]) < 2)
                //    {
                //        NextGeneration.Add(Breed(GenerationMembers[iter], GenerationMembers[iter2]));
                //    }
                //    else
                //    {
                //        NextGeneration.Add(new NeuralNet(NeuralLayerInfo));
                //    }
                //}

                if (iter < SurvivorsPerGeneration * SurvivorsPerGeneration)
                {
                    //for (int iter2 = iter + 1; iter2 < SurvivorsPerGeneration * SurvivorsPerGeneration; iter2 += 1)
                    //{
                        //if (GenerationMembers[iter].DistanceTo(GenerationMembers[iter2]) < 1)
                        //{
                            NextGeneration.Add(Breed(GenerationMembers[Math.Min(iter / SurvivorsPerGeneration, GenerationMembers.Count())],
                                                     GenerationMembers[Math.Min(iter % SurvivorsPerGeneration, GenerationMembers.Count())]));
                        //}
                        //else
                        //{

                        //}
                    //}

                }
                else
                {
                    NextGeneration.Add(Breed(GenerationMembers[GlobalRandom.Next(SurvivorsPerGeneration, NetsPerGeneration)],
                                             GenerationMembers[GlobalRandom.Next(SurvivorsPerGeneration, NetsPerGeneration)]));
                }
            }

            while (NextGeneration.Count() < NetsPerGeneration)
            {
                NextGeneration.Add(new NeuralNet(NeuralLayerInfo));
            }

            GenerationMembers.Clear();
            GenerationMembers = new List<NeuralNet>(NextGeneration);
        }


        internal static NeuralNet Breed(NeuralNet Father, NeuralNet Mother)
        {
            NeuralNet Child = new NeuralNet(NeuralLayerInfo);

            for (UInt16 LayerIter = 0; LayerIter < NeuralLayerInfo.Count(); LayerIter += 1)
            {
                NeuralLayer CurrentLayer = Child.NeuralLayers[LayerIter];

                for (int NeuronIter = 0; NeuronIter < CurrentLayer.NeuronsInLayer.Count(); NeuronIter += 1)
                {
                    Neuron CurrentNeuron = CurrentLayer.NeuronsInLayer[NeuronIter];

                    for (int DendriteIter = 0; DendriteIter < CurrentNeuron.OutputConnections.Count(); DendriteIter += 1)
                    {
                        Dendrite CurrentDendrite = CurrentNeuron.OutputConnections[DendriteIter];

                        if (GlobalRandom.NextDouble() < Math.Min(MaxMutationRate, (BaseMutationRate + (MutationRateIncreasePerFailedGeneration * GensSinceLastImprv))))
                        {
                            CurrentDendrite.SetNewRandomConnectionStrength();
                        }
                        else
                        {
                            // Children are always random
                            if (BreedingType == EnumBreedingType.AlwaysRandom)
                            {
                                CurrentDendrite.SetNewRandomConnectionStrength();
                            }
                            // Child dendrite connection strength is the average of the mother and father
                            else
                            if (BreedingType == EnumBreedingType.AverageValue)
                            {
                                CurrentDendrite.ConnectionStrength =
                                    (Father.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength +
                                        Mother.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength) / 2;
                            }
                            // Child takes fathers or mothers dendrite connection strength
                            else
                            if (BreedingType == EnumBreedingType.Human)
                            {
                                CurrentDendrite.ConnectionStrength = (GlobalRandom.NextDouble() < 0.5)
                                ? Father.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength
                                : Mother.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength;
                            }
                            // Child dendrite connection strength pulled up and down by agreement between father and mother
                            else
                            if (BreedingType == EnumBreedingType.WeightedPull)
                            {
                                if (Father.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength >= 0.5
                                  && Mother.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength >= 0.5)
                                {
                                    CurrentDendrite.ConnectionStrength = (float)Math.Min(1.0, Math.Max(
                                                                          Father.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength,
                                                                          Mother.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength
                                                                          ) + 0.01);
                                }
                                else
                                if (Father.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength < 0.5
                                  && Mother.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength < 0.5)
                                {
                                    CurrentDendrite.ConnectionStrength = (float)Math.Max(0, Math.Min(
                                                                          Father.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength,
                                                                          Mother.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength
                                                                          ) - 0.01);
                                }
                                else
                                {
                                    CurrentDendrite.ConnectionStrength = (GlobalRandom.NextDouble() < 0.5)
                                        ? Father.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength
                                        : Mother.NeuralLayers[LayerIter].NeuronsInLayer[NeuronIter].OutputConnections[DendriteIter].ConnectionStrength;
                                }
                            }
                        }
                    }
                }
            }

            return Child;
        }


        internal static void DisplayGenerationStats(int Generation, NeuralNet BestInGeneration, TimeSpan PropogationTime, TimeSpan NonPropogationTime)
        {
            int iter = 0;
            Console.WriteLine(  "Time:                 {" + iter++ + "}\n" +
                                "Propogation Time:     {" + iter++ + "}\n" +
                                "Meddling Time:        {" + iter++ + "}\n" +
                                "Generation:           {" + iter++ + "}\n" +
                                "Top Score:            {" + iter++ + "} / {" + iter++ + "}\n" +
                                "Sum Of Custom Error:  {" + iter++ + "}\n" +
                                "Sum Of Roots Error:   {" + iter++ + "}\n" +
                                "Sum Of Squares Error: {" + iter++ + "}\n" +
                                "Gens Since Imprv:     {" + iter++ + "}\n" +
                                "\n",
                                TotalTimer.Elapsed,
                                PropogationTime,
                                NonPropogationTime,
                                Generation,
                                BestInGeneration.TotalScore, NumberOfTrainingDatumToUse,
                                BestInGeneration.SumOfCustomError,
                                BestInGeneration.SumOfRootsError,
                                BestInGeneration.SumOfSquaresError,
                                GensSinceLastImprv);
        }
    }
}
