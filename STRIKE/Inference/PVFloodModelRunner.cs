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

public static class PVFloodModelRunner
{
    public static string[] GetNames()
    {
        return ["Rainfall Intensity", "Impervous Ratio", "Drainage Density", "Urbanization Index", "Convergence Index"];
    }
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🌧️ PLUVIAL FLOOD LIKELY | {prob:F4}",
            < 0.44f => $"🛡️ Pluvial Flood Unlikely | {prob:F4}",
            _      => $"⚠️ Pluvial Flood Possible | {prob:F4}"
        };
    }

    private static bool _initialized = false;

    public static string Predict(float[] inputFeatures, bool useTrust, bool net)
    {
        if (net)
        {
            InferenceRunner.UseRemoteInference("PV-Flood");
        }
        else if (!_initialized)
        {
            InferenceRunner.UseLocalInference();
            var scaler = ScalerLoader.Load("Models/scalers/PV-FloodScaler.txt");
            InferenceRunner.Initialize("Models/PV-FloodNet.onnx", "Models/PV-FloodTrustNet.onnx", scaler);
            _initialized = true;
        }

        var inputDict = GetNames().Zip(inputFeatures, (name, value) => new KeyValuePair<string, float>(name, value))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        string label = Eval.GetLabel(adjustedProb);
        SysIO.SavePrediction("🌧️ Pluvial Flood", InferenceRunner.RenameLabel(label), adjustedProb, useTrust, inputDict);
        return label;
    }
}
