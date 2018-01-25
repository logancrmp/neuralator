using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Neuralator
{
    /// <summary>
    /// A Neuron is the individual node that is combined to make up a layer in the neural network.
    /// Each Neuron will have a connection (through a Dendrite) to all Neurons in the previous
    /// layer, as well as all Neurons in the next layer.
    /// </summary>
    internal sealed class Neuron : IEquatable<Neuron>
    {
        /// <summary>
        /// NeuralLayer that this Neuron belongs to.
        /// </summary>
        internal NeuralLayer ParentNeuralLayer { get; private set; }

        /// <summary>
        /// Connections to all Neurons in the previous layer.
        /// This Neuron will set its NeuronValue based on the connections from this layer.
        /// </summary>
        internal List<Dendrite> InputConnections { get; private set; }

        /// <summary>
        /// Connections to all Neurons in the next layer.
        /// This Neuron will propogate its NeuronValue to the connections in the next layer.
        /// </summary>
        internal List<Dendrite> OutputConnections { get; private set; }

        /// <summary>
        /// The current signal strength of this particular Neuron. The NeuronValue will
        /// combine with the connection strength of each output Dendrite to propogate this
        /// value to the Neurons in the next layer.
        /// </summary>
        internal float NeuronValue { get; private set; }

        /// <summary>
        /// Unique ID to be used for ensuring unique hash codes for Dendrites, but equal across idential NeuralNets.
        /// </summary>
        internal UInt32 UniqueID { get; private set; }


        /// <summary>
        /// Construct an instance of a Neuron. This should only be called from the
        /// constructor of the NeuralLayer that will own this Neuron.
        /// </summary>
        /// <param name="ParentNeuralLayer">A reference to the parent NeuralLayer that constructed this Neuron.</param>
        internal Neuron(NeuralLayer ParentNeuralLayer, UInt32 UniqueID)
        {
            this.UniqueID = UniqueID;

            InputConnections = new List<Dendrite>();
            OutputConnections = new List<Dendrite>();

            this.ParentNeuralLayer = ParentNeuralLayer;
        }

        /// <summary>
        /// Connect this Neuron to each of the Neurons in the next layer.
        /// </summary>
        internal void Connect()
        {
            NeuralLayer NextNeuralLayer = ParentNeuralLayer.GetNextNeuralLayer();

            if (NextNeuralLayer != default(NeuralLayer))
            {
                foreach (var NeuronToConnect in NextNeuralLayer.NeuronsInLayer)
                {
                    OutputConnections.Add(new Dendrite(this, NeuronToConnect));
                }
            }
        }


        /// <summary>
        /// Add the given Dendrite to the list of input connections.
        /// </summary>
        /// <param name="Connection"></param>
        internal void AddInputConnection(Dendrite Connection)
        {
            InputConnections.Add(Connection);
        }


        /// <summary>
        /// Add the given Dendrite to the list of output connections.
        /// </summary>
        /// <param name="Connection"></param>
        internal void AddOutputConnection(Dendrite Connection)
        {
            OutputConnections.Add(Connection);
        }


        /// <summary>
        /// Propogate the value of the Neuron to each of the Neurons in the next layer
        /// through the output Dendrites.
        /// </summary>
        internal void Propogate()
        {
            foreach (var Dendrite in OutputConnections)
            {
                Dendrite.Propogate();
            }
        }


        /// <summary>
        /// Signal the Neuron with an incoming value (coming from an input Neuron in the previous layer).
        /// This will scale the value down based on the number of expected valid input connections, then
        /// adds the scaled value to the current NeuronValue.
        /// </summary>
        /// <param name="InputSignal"></param>
        internal void SignalNeuron(float InputSignal)
        {
            InputSignal /= (InputConnections.Count() * (float)Math.Pow(Neuralator.DendriteFillRate, ParentNeuralLayer.LayerIndex));

            NeuronValue = Math.Min(NeuronValue + InputSignal, 1.0f);
        }


        /// <summary>
        /// Sets the initial value of the Neuron. This should only be used to set the
        /// inputs to the first NeuralLayer.
        /// </summary>
        /// <param name="Value"></param>
        internal void SetInitialValue(float Value)
        {
            NeuronValue = Value;
        }


        /// <summary>
        /// Resets the Neuron to a default (but still connected via Dendrites) state for reuse.
        /// </summary>
        internal void ClearNeuronState()
        {
            NeuronValue = 0;
        }

        public bool Equals(Neuron other)
        {
            return (GetDimensionMap() == other.GetDimensionMap());
        }


        internal Dictionary<ulong, double> GetDimensionMap()
        {
            Dictionary<ulong, double> DimensionMap = new Dictionary<ulong, double>();

            foreach (var Connection in OutputConnections)
            {
                DimensionMap.Add(Connection.UniqueDendriteID, Connection.ConnectionStrength);
            }

            return DimensionMap;
        }
    }
}
