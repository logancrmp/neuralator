using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;

namespace Neuralator
{
    internal class NeuralNet : IComparable<NeuralNet>
    {
        internal Dictionary<UInt16, NeuralLayer> NeuralLayers { get; private set; }

        internal Dictionary<ulong, double> DimensionMap;

        internal int OutputValue { get; private set; }
        internal float Confidence { get; private set; }

        internal int TotalScore;
        internal double TotalError;
        internal double SumOfSquaresError;
        internal double SumOfRootsError;
        internal double SumOfCustomError;
        internal float TotalAvgConfidence;


        /// <summary>
        /// Populate NeuralNet with all layers, neurons and connections
        /// </summary>
        /// <param name="NeuralLayerInfo">Each entry in the list is number of neurons in that layer. One layer per entry.</param>
        /// <param name="DendriteFillRate">Percent of all neurons in the next layer that a given neuron will connect to. 0.0 to 1.0</param>
        internal NeuralNet(List<UInt16> NeuralLayerInfo)
        {
            NeuralLayers = new Dictionary<UInt16, NeuralLayer>();

            for (UInt16 iter = 0; iter < NeuralLayerInfo.Count(); iter += 1)
            {
                NeuralLayers.Add(iter, new NeuralLayer(this, iter, NeuralLayerInfo[iter]));
            }

            for (UInt16 iter = 0; iter < NeuralLayers.Count() - 1; iter += 1)
            {
                NeuralLayers[iter].ConnectToNextLayer();
            }

            // DimensionMap = GetDimensionMap();
        }


        internal void ClearScore()
        {
            TotalScore = 0;
            TotalError = 0;
            TotalAvgConfidence = 0;

            SumOfRootsError = 0;
            SumOfSquaresError = 0;
            SumOfCustomError = 0;
        }


        internal void ClearNetworkState()
        {
            OutputValue = 0;
            Confidence = 0;

            foreach (var Layer in NeuralLayers.Values)
            {
                Layer.ClearLayerState();
            }
        }


        //Make this a delagate for adding a float to n entries or something?
        internal void SetInputsToFirstLayer(TrainingDatum CurrentTest)
        {
            for (int iter = 0; iter < NeuralLayers[0].NeuronsInLayer.Count(); iter += 1)
            {
                NeuralLayers[0].NeuronsInLayer[iter].SetInitialValue(CurrentTest.InputLayerValues.ElementAt(Math.Min(iter, CurrentTest.InputLayerValues.Count - 1)));
            }
        }


        internal void Propogate()
        {
            foreach (var Layer in NeuralLayers.Values)
            {
                Layer.Propogate();
            }
        }


        internal void GetConsensus()
        {
            var FinalLayerNeurons = NeuralLayers[(UInt16)(NeuralLayers.Count() - 1)].NeuronsInLayer;

            Confidence = int.MinValue;

            for (UInt16 iter = 0; iter < FinalLayerNeurons.Count(); iter += 1)
            {
                var FinalNeuron = FinalLayerNeurons[iter];

                if (FinalNeuron.NeuronValue > Confidence)
                {
                    OutputValue = iter;
                    Confidence = FinalNeuron.NeuronValue;
                }
            }
        }


        internal Dictionary<ulong, double> GetDimensionMap()
        {
            Dictionary<ulong, double> DimensionMap = new Dictionary<ulong, double>();

            foreach (var Layer in NeuralLayers.Values)
            {
                foreach (var Dimension in Layer.GetDimensionMap())
                {
                    DimensionMap.Add(Dimension.Key, Dimension.Value);
                }
                
            }

            return DimensionMap;
        }


        internal double DistanceTo(NeuralNet Other)
        {
            double Distance = 0;

            foreach (var Key in DimensionMap.Keys)
            {
                Distance += Math.Pow(DimensionMap[Key] - Other.DimensionMap[Key], 2);
            }
            
            return Math.Sqrt(Distance);
        }


        public int CompareTo(NeuralNet Other)
        {
            int ReturnValue = 0;

            if (Neuralator.SortByScore)
            {
                if (TotalScore > Other.TotalScore)
                {
                    ReturnValue = -1;
                }
                else
                if (TotalScore < Other.TotalScore)
                {
                    ReturnValue = 1;
                }
                else
                {
                    if (TotalError < Other.TotalError)
                    {
                        ReturnValue = -1;
                    }
                    else
                    if (TotalError > Other.TotalError)
                    {
                        ReturnValue = 1;
                    }
                    else
                    {
                        if (TotalAvgConfidence > Other.TotalAvgConfidence)
                        {
                            ReturnValue = -1;
                        }
                        else
                        if (TotalAvgConfidence < Other.TotalAvgConfidence)
                        {
                            ReturnValue = 1;
                        }
                    }
                }
            }
            else
            {
                if (TotalError < Other.TotalError)
                {
                    ReturnValue = -1;
                }
                else
                if (TotalError > Other.TotalError)
                {
                    ReturnValue = 1;
                }
                else
                {
                    if (TotalScore > Other.TotalScore)
                    {
                        ReturnValue = -1;
                    }
                    else
                    if (TotalScore < Other.TotalScore)
                    {
                        ReturnValue = 1;
                    }
                    else
                    {
                        if (TotalAvgConfidence > Other.TotalAvgConfidence)
                        {
                            ReturnValue = -1;
                        }
                        else
                        if (TotalAvgConfidence < Other.TotalAvgConfidence)
                        {
                            ReturnValue = 1;
                        }
                    }
                }
            }

            return ReturnValue;
        }
    }
}
