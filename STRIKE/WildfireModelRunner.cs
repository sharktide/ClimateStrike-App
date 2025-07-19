using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;

public static class WildfireModelRunner
{
    private static readonly InferenceSession _baseSession = new("Models/FireNet.onnx");
    private static readonly InferenceSession _trustSession = new("Models/FireTrustNet.onnx");

    private static readonly float[] _mean;
    private static readonly float[] _scale;

    static WildfireModelRunner()
    {
        try
        {
            var scalerJson = File.ReadAllText("Models/scalers/FireScaler.json");
            var scaler = JsonSerializer.Deserialize<StandardScaler>(scalerJson);
            _mean = scaler.mean.Select(x => (float)x).ToArray();
            _scale = scaler.scale.Select(x => (float)x).ToArray();

        }
        catch (Exception ex)
        {
            Console.WriteLine($"🔥 Failed to load scaler: {ex.Message}");
            _mean = new float[5];
            _scale = new float[5];
        }
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        // Step 1: Base model prediction
        var baseTensor = new DenseTensor<float>(inputFeatures, new[] { 1, inputFeatures.Length });
        var baseInputName = _baseSession.InputMetadata.Keys.First();
        var baseOutputName = _baseSession.OutputMetadata.Keys.First();

        var baseResults = _baseSession.Run(new[] {
            NamedOnnxValue.CreateFromTensor(baseInputName, baseTensor)
        });

        var baseProb = baseResults.First().AsEnumerable<float>().First();

        // Step 2: TrustNet modulation
        float adjustedProb = baseProb;

        if (useTrust)
        {
            var scaled = Standardize(inputFeatures);
            var trustTensor = new DenseTensor<float>(scaled, new[] { 1, scaled.Length });
            var trustInputName = _trustSession.InputMetadata.Keys.First();
            var trustOutputName = _trustSession.OutputMetadata.Keys.First();

            var trustResults = _trustSession.Run(new[] {
                NamedOnnxValue.CreateFromTensor(trustInputName, trustTensor)
            });

            var trustFactor = trustResults.First().AsEnumerable<float>().First();
            adjustedProb = Math.Clamp(baseProb * trustFactor, 0, 1);
        }

        if ((adjustedProb) > 0.49)
        {
            return $"🔥 FIRE LIKELY | {adjustedProb:F4}";
        }
        else if ((adjustedProb) < 0.44)
        {
            return $"🛡️ Fire Unlikely | {adjustedProb:F4}";
        }
        else
        {
            return $"⚠️ Fire Possible | {adjustedProb:F4}";
        }
    }

    private static float[] Standardize(float[] input)
    {
        var scaled = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            scaled[i] = (input[i] - _mean[i]) / _scale[i];
        }
        return scaled;
    }

    private class StandardScaler
    {
        public List<double> mean { get; set; }
        public List<double> scale { get; set; }
    }

}
