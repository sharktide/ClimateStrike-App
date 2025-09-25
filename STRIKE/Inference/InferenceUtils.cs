namespace STRIKE.InferenceUtils;
#pragma warning disable CS0618
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;
using System.IO;
using Microsoft.Maui.Storage;
using Microsoft.Maui.Controls;

public static class ScalerLoader
{
    private static readonly JsonSerializerOptions CachedJsonOptions = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };
    public static StandardScaler Load(string relativePath)
    {
        var fullPath = Path.Combine(AppContext.BaseDirectory, relativePath);
        var json = File.ReadAllText(fullPath);

        return JsonSerializer.Deserialize<StandardScaler>(json, CachedJsonOptions)
               ?? throw new InvalidOperationException("Failed to load scaler.");
    }
}

public static class InferenceRunner
{
    private static InferenceSession? _baseSession;
    private static InferenceSession? _trustSession;
    private static StandardScaler? _scaler;

    public static void Initialize(string baseModelFileName, string trustModelFileName, StandardScaler scaler)
    {
        try
        {
            _scaler = scaler;

            var baseModelPath = Path.Combine(AppContext.BaseDirectory, baseModelFileName);
            var trustModelPath = Path.Combine(AppContext.BaseDirectory, trustModelFileName);

            _baseSession = new InferenceSession(baseModelPath);
            _trustSession = new InferenceSession(trustModelPath);
        }
        catch (Exception ex)
        {
            Exception current = ex;
            while (current.InnerException != null)
            {
                current = current.InnerException;
                var rootCause = $"Type: {current.GetType().Name} Message: {current.Message} Trace: {current.StackTrace}";
            }
        }
    }

    public static float RunPrediction(float[] inputFeatures, bool useTrust)
    {
        var baseTensor = new DenseTensor<float>(inputFeatures, new[] { 1, inputFeatures.Length });
        var inputName = _baseSession!.InputMetadata.Keys.First();

        var baseResults = _baseSession.Run(new[] {
            NamedOnnxValue.CreateFromTensor(inputName, baseTensor)
        });

        var baseProb = baseResults.First().AsEnumerable<float>().First();

        float adjustedProb = baseProb;

        if (useTrust && _scaler != null)
        {
            var scaled = _scaler.Transform(inputFeatures);
            var trustTensor = new DenseTensor<float>(scaled, new[] { 1, scaled.Length });
            var trustInputName = _trustSession!.InputMetadata.Keys.First();

            var trustResults = _trustSession.Run(new[] {
                NamedOnnxValue.CreateFromTensor(trustInputName, trustTensor)
            });

            var trustFactor = trustResults.First().AsEnumerable<float>().First();
            adjustedProb = Math.Clamp(baseProb * trustFactor, 0, 1);
        }

        return adjustedProb;
    }

    public static string RenameLabel(string label)
    {
        if (label.Contains("LIKELY"))
        {
            return "High Risk";
        }
        else if (label.Contains("Possible"))
        {
            return "Medium Risk";
        }
        else
        {
            return "Low Risk";
        }
    }
    public static readonly Dictionary<string, string> FeatureUnits = new()
    {
        // Earthquake
        { "Seismic Moment Rate", "×10¹⁶ Nm/s" },
        { "Surface Displacement Rate", "mm/yr" },
        { "Coulomb Stress Change", "Pa" },
        { "Average Focal Depth", "km" },
        { "Fault Slip Rate", "mm/yr" },

        // Flash Flood
        { "Rainfall Intensity", "mm" },
        { "Slope", "°" },
        { "Drainage Density", "s" },
        { "Soil Saturation", "m" },
        { "Convergence Index", "m" },

        // Fluvial Flood
        { "Rainfall", "mm" },
        { "River Water Level", "mm" },
        { "Relative Slope", "s" },
        { "Elevation", "m" },
        { "Distance to River", "m" },

        // Pluvial Flood
        { "Rainfall Intensity (Pluvial)", "mm" },
        { "Imperviousness", "mm" },
        { "Drainage Density (Pluvial)", "s" },
        { "Urbanization Index", "m" },
        { "Convergence Index (Pluvial)", "m" },

        // Hurricane
        { "Sea Surface Temperature", "°C/°F" },
        { "Ocean Heat Content", "kJ/cm²" },
        { "Mid-Level Humidity", "%" },
        { "Vertical Wind Shear", "m/s" },
        { "Potential Vorticity", "PVU" },

        // Tornado
        { "Storm Relative Helicity (SRH)", "m²/s²" },
        { "CAPE", "J/Kg" },
        { "Lifted Condensation Level (LCL)", "m" },
        { "Bulk Wind Shear", "m/s" },
        { "Significant Tornado Parameter (STP)", "STP" },

        // Wildfire
        { "Temperature", "K" },
        { "Humidity (Wildfire)", "%" },
        { "Wind Speed", "m/s" },
        { "Vegetation Index", "NDVI" },
    };

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



public static class SysIO
{
    private static readonly JsonSerializerOptions CachedJsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };
    public static readonly string appDataPath = FileSystem.AppDataDirectory;
    public static readonly string recentPredictions = Path.Combine(appDataPath ?? "./", "predictions.json");

    public static string SavePrediction(string disasterType, string label, float probability, bool trust, Dictionary<string, float> inputs)
    {
        var prediction = new STRIKE.Services.PredictionResult
        {
            DisasterType = disasterType,
            Label = label,
            Probability = probability,
            Trust = trust,
            Inputs = inputs,
            Timestamp = DateTime.Now
        };

        List<STRIKE.Services.PredictionResult> history;

        try
        {
            if (File.Exists(recentPredictions))
            {
                var json = File.ReadAllText(recentPredictions);
                history = JsonSerializer.Deserialize<List<STRIKE.Services.PredictionResult>>(json) ?? [];
            }
            else
            {
                history = [];
            }
        }
        catch
        {
            history = [];
        }

        history.Insert(0, prediction);

        var outJson = JsonSerializer.Serialize(history, CachedJsonOptions);
        File.WriteAllText(recentPredictions, outJson);
        return $"written text to {recentPredictions}. Data: {outJson}";
    }
    public static List<STRIKE.Services.PredictionResult> GetPredictionHistory()
    {
        if (File.Exists(recentPredictions))
        {
            var json = File.ReadAllText(recentPredictions);
            var history = JsonSerializer.Deserialize<List<STRIKE.Services.PredictionResult>>(json, CachedJsonOptions);
            return history ?? [];
        }
        else
        {
            return [];
        }
    }
    public static void DeletePrediction(DateTime timestamp)
    {
        if (!File.Exists(recentPredictions)) return;
        var json = File.ReadAllText(recentPredictions);
        var history = JsonSerializer.Deserialize<List<STRIKE.Services.PredictionResult>>(json, CachedJsonOptions) ?? [];
        var newHistory = history.Where(p => p.Timestamp != timestamp).ToList();
        var outJson = JsonSerializer.Serialize(newHistory, CachedJsonOptions);
        File.WriteAllText(recentPredictions, outJson);
    }

    public static void DeleteAllPredictions()
    {
        if (File.Exists(recentPredictions))
        {
            File.Delete(recentPredictions);
        }
    }

}