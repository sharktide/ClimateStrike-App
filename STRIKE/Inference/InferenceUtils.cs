namespace STRIKE.InferenceUtils;

using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;

public static class ScalerLoader
{
    public static StandardScaler Load(string path)
    {
        var json = File.ReadAllText(path);
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };

        return JsonSerializer.Deserialize<StandardScaler>(json, options)
            ?? throw new InvalidOperationException("Failed to load scaler.");
    }
}

public static class InferenceRunner
{
    private static InferenceSession _baseSession;
    private static InferenceSession _trustSession;
    private static StandardScaler? _scaler;

    public static void Initialize(string baseModelPath, string trustModelPath, StandardScaler scaler)
    {
        _baseSession = new InferenceSession(baseModelPath);
        _trustSession = new InferenceSession(trustModelPath);
        _scaler = scaler;
    }

    public static float RunPrediction(float[] inputFeatures, bool useTrust)
    {
        var baseTensor = new DenseTensor<float>(inputFeatures, new[] { 1, inputFeatures.Length });
        var inputName = _baseSession.InputMetadata.Keys.First();

        var baseResults = _baseSession.Run(new[] {
            NamedOnnxValue.CreateFromTensor(inputName, baseTensor)
        });

        var baseProb = baseResults.First().AsEnumerable<float>().First();

        float adjustedProb = baseProb;

        if (useTrust && _scaler != null)
        {
            var scaled = _scaler.Transform(inputFeatures);
            var trustTensor = new DenseTensor<float>(scaled, new[] { 1, scaled.Length });
            var trustInputName = _trustSession.InputMetadata.Keys.First();

            var trustResults = _trustSession.Run(new[] {
                NamedOnnxValue.CreateFromTensor(trustInputName, trustTensor)
            });

            var trustFactor = trustResults.First().AsEnumerable<float>().First();
            adjustedProb = Math.Clamp(baseProb * trustFactor, 0, 1);
        }

        return adjustedProb;
    }
}


public class StandardScaler
{

    [JsonConstructor]
    public StandardScaler(List<double> mean, List<double> scale)
    {
        Mean = mean;
        Scale = scale;
    }
    public List<double> Mean { get; set; }
    public List<double> Scale { get; set; }

    public float[] Transform(float[] input)
    {
        var scaled = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            scaled[i] = (input[i] - (float)Mean[i]) / (float)Scale[i];
        }
        return scaled;
    }
}
