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

namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class EarthquakeModelRunner
{
    public static string[] GetNames()
    {
        return ["Seismic Moment Rate", "Surface Displacement Rate", "Coulomb Stress Change", "Average Focal Depth", "Fault Slip Rate"];
    }
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🌎 EARTHQUAKE LIKELY  | {prob:F4}",
            < 0.44f => $"🛡️ Earthquake Unlikely | {prob:F4}",
            _      => $"⚠️ Earthquake Possible | {prob:F4}"
        };
    }

    private static bool _initialized = false;

    public static string Predict(float[] inputFeatures, bool useTrust, bool net)
    {
        if (net)
        {
            InferenceRunner.UseRemoteInference("Earthquake");
        }
        else if (!_initialized)
        {
            InferenceRunner.UseLocalInference();
            var scaler = ScalerLoader.Load("Models/scalers/QuakeScaler.txt");
            InferenceRunner.Initialize("Models/QuakeNet.onnx", "Models/QuakeTrustNet.onnx", scaler);
            _initialized = true;
        }

        var inputDict = GetNames().Zip(inputFeatures, (name, value) => new KeyValuePair<string, float>(name, value))
            .ToDictionary(pair => pair.Key, pair => pair.Value);

        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        string label = Eval.GetLabel(adjustedProb);
        SysIO.SavePrediction("🌎 Earthquake", InferenceRunner.RenameLabel(label), adjustedProb, useTrust, inputDict);
        return label;
    }
}
