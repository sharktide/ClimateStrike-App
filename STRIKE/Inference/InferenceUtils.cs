/* Copyright 2025 Rihaan Meher

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

namespace STRIKE.InferenceUtils;
#pragma warning disable CS0618
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.IO;
using Microsoft.Maui.Storage;
using Microsoft.Maui.Controls;
using Microsoft.Maui.Devices;
using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Net.Http.Json;
using System.Text.RegularExpressions;

using System.Text.Json;
using System.Text.Json.Serialization;
using STRIKE.Services;

[JsonSourceGenerationOptions(WriteIndented = true, PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(List<PredictionResult>))]
internal partial class PredictionJsonContext : JsonSerializerContext
{
}
public static class ScalerLoader
{
    private static readonly JsonSerializerOptions CachedJsonOptions = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };
    public static StandardScaler Load(string relativePath)
    {
        string? json = null;

        try
        {
            var fullPath = Path.Combine(AppContext.BaseDirectory, relativePath);
            if (File.Exists(fullPath))
            {
                json = File.ReadAllText(fullPath);
            }
        }
        catch { }

        if (json == null)
        {
            try
            {
                var fileName = Path.GetFileName(relativePath);
                using var stream = Microsoft.Maui.Storage.FileSystem.OpenAppPackageFileAsync(fileName).GetAwaiter().GetResult();
                using var reader = new StreamReader(stream);
                json = reader.ReadToEnd();
            }
            catch
            {
                try
                {
                    using var stream = Microsoft.Maui.Storage.FileSystem.OpenAppPackageFileAsync(relativePath).GetAwaiter().GetResult();
                    using var reader = new StreamReader(stream);
                    json = reader.ReadToEnd();
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Failed to load scaler from '{relativePath}' (file system and app package attempts failed)", ex);
                }
            }
        }

        return JsonSerializer.Deserialize<StandardScaler>(json!, CachedJsonOptions)
               ?? throw new InvalidOperationException("Failed to load scaler.");
    }
}

public static partial class InferenceRunner
{
    public static bool _useRemoteInference = false;
    private static string _remoteEndpoint = "https://sharktide-strike.hf.space/api/predict/";
    private static string _currentModel = "Wildfire";

    private static InferenceSession? _baseSession;
    private static InferenceSession? _trustSession;
    private static StandardScaler? _scaler;
    private static string? _baseModelPath;
    private static string? _trustModelPath;

    public static void Initialize(string baseModelFileName, string trustModelFileName, StandardScaler scaler)
    {
        try
        {
            _scaler = scaler;

            _baseModelPath = Path.Combine(AppContext.BaseDirectory, baseModelFileName);
            _trustModelPath = Path.Combine(AppContext.BaseDirectory, trustModelFileName);

            var platform = DeviceInfo.Platform;
            if (!_useRemoteInference && platform != DevicePlatform.MacCatalyst && platform != DevicePlatform.iOS)
            {
                _baseSession = new InferenceSession(_baseModelPath);
            }

                _trustSession = new InferenceSession(_trustModelPath);
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

    private static bool TryInitializeBaseSessionIfAllowed()
    {
        try
        {
            if (_baseSession != null) return true;

            if (_useRemoteInference) return false;

            var platform = DeviceInfo.Platform;
            if (platform == DevicePlatform.MacCatalyst || platform == DevicePlatform.iOS) return false;

            if (string.IsNullOrEmpty(_baseModelPath)) return false;

            _baseSession = new InferenceSession(_baseModelPath);
            return true;
        }
        catch
        {
            return false;
        }
    }

    public static void UseRemoteInference(string modelName)
    {
        _useRemoteInference = true;
        _currentModel = modelName;
    }
    public static void UseLocalInference()
    {
        _useRemoteInference = false;
    }

    public static float RunPrediction(float[] inputFeatures, bool useTrust, string extras = "", bool isWildfire = false)
    {
        if (_useRemoteInference)
        {
            if (!NetHelper.IsConnected())
            {
                // Try to lazily initialize the base session if the app was allowed to load it
                var initialized = TryInitializeBaseSessionIfAllowed();
                if (initialized)
                {
                    goto LocalInference;
                }

                throw new InvalidOperationException("Remote inference requested but no network connectivity and local models are not initialized or not allowed on this platform.");
            }

            try
            {
                string query;
                using var client = new HttpClient();
                if (isWildfire)
                {
                    query = $"{_remoteEndpoint}{_currentModel}?" +
                            $"param1={inputFeatures[0]}&param2={inputFeatures[1]}&param3={inputFeatures[2]}" +
                            $"&param4={inputFeatures[4]}&param5={inputFeatures[3]}&use_trust={useTrust.ToString().ToLower()}{extras}";
                }
                else
                {
                    query = $"{_remoteEndpoint}{_currentModel}?" +
                            $"param1={inputFeatures[0]}&param2={inputFeatures[1]}&param3={inputFeatures[2]}" +
                            $"&param4={inputFeatures[3]}&param5={inputFeatures[4]}&use_trust={useTrust.ToString().ToLower()}{extras}";
                }

                var response = client.GetAsync(query).GetAwaiter().GetResult();
                response.EnsureSuccessStatusCode();

                var json = response.Content.ReadFromJsonAsync<RemoteResponse>().GetAwaiter().GetResult();

                if (json?.Result is string resultText)
                {
                    var match = MyRegex().Match(resultText);
                    if (match.Success && float.TryParse(match.Groups[1].Value, out float parsedProb))
                    {
                        return parsedProb;
                    }
                    else
                    {
                        throw new JsonException("Unable to parse probability from remote response");
                    }
                }

                throw new InvalidOperationException("Remote response was null or malformed.");
            }
            catch (Exception ex)
            {
                // If remote fails, but local models exist, fall back to local inference rather than throwing.
                // If remote fails, attempt to lazily create base session if allowed and fall back
                var initialized = TryInitializeBaseSessionIfAllowed();
                if (initialized)
                {
                    try
                    {
                        goto LocalInference;
                    }
                    catch { /* fall through to error below if local also fails */ }
                }

                throw new InvalidOperationException("Remote inference failed and no local fallback available.", ex);
            }
        }
        
        LocalInference:
        if (_baseSession == null)
        {
            throw new InvalidOperationException("Local model session is not initialized. Call Initialize(...) before running local inference or enable remote inference when network is available.");
        }

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
            var trustInputName = _trustSession!.InputMetadata.Keys.First();

            var trustResults = _trustSession.Run(new[] {
                NamedOnnxValue.CreateFromTensor(trustInputName, trustTensor)
            });

            var trustFactor = trustResults.First().AsEnumerable<float>().First();
            adjustedProb = Math.Clamp(baseProb * trustFactor, 0, 1);
        }

        return adjustedProb;
    }

    private class RemoteResponse
    {
        public required string Result { get; set; }
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

    [System.Text.RegularExpressions.GeneratedRegex(@"\(([\d.]+)\)")]
    private static partial System.Text.RegularExpressions.Regex MyRegex();
}

public static class NetHelper
{
    public static bool IsConnected()
    {
        return Connectivity.Current.NetworkAccess == NetworkAccess.Internet;
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

public static class SysIO
{
    public static readonly string appDataPath = FileSystem.AppDataDirectory;
    public static readonly string recentPredictions = Path.Combine(appDataPath ?? "./", "predictions.json");

    public static string SavePrediction(string disasterType, string label, float probability, bool trust, Dictionary<string, float> inputs)
    {
        var prediction = new PredictionResult
        {
            DisasterType = disasterType,
            Label = label,
            Probability = probability,
            Trust = trust,
            Inputs = inputs,
            Timestamp = DateTime.Now
        };

        List<PredictionResult> history;

        try
        {
            if (File.Exists(recentPredictions))
            {
                var json = File.ReadAllText(recentPredictions);
                history = JsonSerializer.Deserialize(json, PredictionJsonContext.Default.ListPredictionResult) ?? [];
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

        var outJson = JsonSerializer.Serialize(history, PredictionJsonContext.Default.ListPredictionResult);
        File.WriteAllText(recentPredictions, outJson);
        return $"written text to {recentPredictions}. Data: {outJson}";
    }

    public static List<PredictionResult> GetPredictionHistory()
    {
        if (File.Exists(recentPredictions))
        {
            var json = File.ReadAllText(recentPredictions);
            var history = JsonSerializer.Deserialize(json, PredictionJsonContext.Default.ListPredictionResult);
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
        var history = JsonSerializer.Deserialize(json, PredictionJsonContext.Default.ListPredictionResult) ?? [];
        var newHistory = history.Where(p => p.Timestamp != timestamp).ToList();
        var outJson = JsonSerializer.Serialize(newHistory, PredictionJsonContext.Default.ListPredictionResult);
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
