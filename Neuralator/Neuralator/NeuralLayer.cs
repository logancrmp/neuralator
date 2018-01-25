using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;

namespace Neuralator
{
    /// <summary>
    /// A NeuralNet is made up of a sequence of NeuralLayers. Each NeuralLayer contains a set
    /// of Neurons that will have connections to all of the Neurons in the next layer as well
    /// as all of the Neurons in the previous layer. A NeuralLayer does not really have any
    /// functionality of its own, but is a convenient tool for organizing the NeuralNet.
    /// </summary>
    internal class NeuralLayer
    {
        internal NeuralNet ParentNeuralNet { get; private set; }
        internal UInt16 LayerIndex { get; private set; }
        internal List<Neuron> NeuronsInLayer { get; private set; }


        internal NeuralLayer(NeuralNet ParentNeuralNet, UInt16 LayerIndex, uint NumberOfNeuronsInLayer)
        {
            this.LayerIndex = LayerIndex;
            this.ParentNeuralNet = ParentNeuralNet;

            NeuronsInLayer = new List<Neuron>();

            UInt32 UniqueID = ((UInt32)LayerIndex << 16);
            for (UInt16 iter = 0; iter < NumberOfNeuronsInLayer; iter += 1)
            {
                NeuronsInLayer.Add(new Neuron(this, (UniqueID | iter)));
            }
        }


        internal void Propogate()
        {
            foreach (var Neuron in NeuronsInLayer)
            {
                Neuron.Propogate();
            }
        }


        internal NeuralLayer GetNextNeuralLayer()
        {
            NeuralLayer ReturnValue = default(NeuralLayer);

            if (ParentNeuralNet.NeuralLayers.ContainsKey((UInt16)(LayerIndex + 1)))
            {
                ReturnValue = ParentNeuralNet.NeuralLayers[(UInt16)(LayerIndex + 1)];
            }

            return ReturnValue;
        }


        internal void ConnectToNextLayer()
        {
            foreach (var Neuron in NeuronsInLayer)
            {
                Neuron.Connect();
            }
        }


        internal void ClearLayerState()
        {
            foreach (var Neuron in NeuronsInLayer)
            {
                Neuron.ClearNeuronState();
            }
        }

        internal Dictionary<ulong, double> GetDimensionMap()
        {
            Dictionary<ulong, double> DimensionMap = new Dictionary<ulong, double>();

            foreach (var Neuron in NeuronsInLayer)
            {
                foreach (var Dimension in Neuron.GetDimensionMap())
                {
                    DimensionMap.Add(Dimension.Key, Dimension.Value);
                }
            }

            return DimensionMap;
        }

        public bool Equals(NeuralLayer other)
        {
            bool Equals = (LayerIndex == other.LayerIndex);

            for (int iter = 0; (iter < NeuronsInLayer.Count && Equals); iter += 1)
            {
                Equals &= NeuronsInLayer[iter].Equals(other.NeuronsInLayer[iter]);
            }

            return Equals;
        }
    }
}
