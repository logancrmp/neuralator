using System.Collections.Generic;

namespace Neuralator
{
    class TrainingDatum
    {
        internal readonly float OutputValue;
        internal readonly IReadOnlyCollection<float> InputLayerValues;

        internal TrainingDatum(float OutputValue, List<float> InputLayerValues)
        {
            this.OutputValue = OutputValue;
            this.InputLayerValues = InputLayerValues.AsReadOnly();
        }
    }
}
