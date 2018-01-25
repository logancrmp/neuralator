
using System;
using System.Collections.Generic;

namespace Neuralator
{
    /// <summary>
    /// A Dendrite is the connection between two Neurons. During propogation, it takes a value from
    /// the input Neuron, modifies based on the Dendrites connection strength, and signals the
    /// output Neuron.
    /// </summary>
    internal class Dendrite
    {
        internal Neuron InputNeuron { get; private set; }
        internal Neuron OutputNeuron { get; private set; }

        /// <summary>
        /// How strong the connection between the two Neurons is.
        /// The Dendrite will signal OutputNeuron with the value InputNeuron * ConnectionStrength.
        /// Setting this to 0 effectively kills the connection.
        /// </summary>
        internal float ConnectionStrength;

        /// <summary>
        /// Value used to uniquely identify the Dendrite from all others for calculating disatance
        /// between NeuralNets.
        /// </summary>
        internal UInt64 UniqueDendriteID;


        /// <summary>
        /// Create a new Dendrite connection between two Neurons.
        /// </summary>
        /// <param name="InputNeuron">Input Neuron that will provide the signal to propogate.</param>
        /// <param name="OutputNeuron">Output Neuron that will receive the signal.</param>
        internal Dendrite(Neuron InputNeuron, Neuron OutputNeuron)
        {
            this.InputNeuron = InputNeuron;
            this.OutputNeuron = OutputNeuron;

            UniqueDendriteID = ((UInt64)InputNeuron.UniqueID << 32) | OutputNeuron.UniqueID;

            SetNewRandomConnectionStrength();

            OutputNeuron.AddInputConnection(this);
        }


        /// <summary>
        /// Take the value of the input Neuron and conditionally propogate the value to the output Neuron.
        /// This will multiply the input Neurons value by the strength of the connection, and if it is over
        /// the configurable threshold (minimum action potential), the output Neuron will be signaled.
        /// </summary>
        internal void Propogate()
        {
            var ActualActionPotential = InputNeuron.NeuronValue * ConnectionStrength;

            if (ActualActionPotential > Neuralator.MinimumActionPotential)
            {
                OutputNeuron.SignalNeuron(ActualActionPotential);
            }
        }


        /// <summary>
        /// Set a random connection strength for the Dendrite. This will determine how strongly the
        /// output Neuron is affected by the input Neuron. Based on the configuration parameter DendriteFillRate,
        /// this function can result in a connection strength of 0, which effectively kills the connection
        /// between the input and output Neurons.
        /// </summary>
        internal void SetNewRandomConnectionStrength()
        {
            float NewConnectionStrength = 0f;
            
            if (Neuralator.GlobalRandom.NextDouble() >= (1 - Neuralator.DendriteFillRate))
            {
                // Set to a random value from 0.0 to 1.0
                NewConnectionStrength = (float) Neuralator.GlobalRandom.NextDouble();
                // Scale the value
                NewConnectionStrength *= (Neuralator.MaxConnectionStrength - Neuralator.MinConnectionStrength);
                // Offset the value
                NewConnectionStrength += Neuralator.MinConnectionStrength;
            }

            ConnectionStrength = NewConnectionStrength;
        }
    }
}
