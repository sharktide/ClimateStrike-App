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

public static class FlashFloodModelRunner
{
    public static string[] GetNames()
    {
        return ["Rainfall Intensity", "Terrain Gradient", "Drainage Density", "Soil Saturation", "Convergence Index"];
    }
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🌩️ FLASH FLOOD LIKELY | {prob:F4}",
            < 0.44f => $"🛡️ Flash Flood Unlikely | {prob:F4}",
            _      => $"⚠️ Flash Flood Possible | {prob:F4}"
        };
    }

    static FlashFloodModelRunner()
    {
        var scaler = ScalerLoader.Load("Models/scalers/FlashFloodScaler.txt");

        InferenceRunner.Initialize("Models/FlashFloodNet.onnx", "Models/FlashFloodTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var inputDict = GetNames().Zip(inputFeatures, (name, value) => new KeyValuePair<string, float>(name, value))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        string label = Eval.GetLabel(adjustedProb);
        SysIO.SavePrediction("🌩️ Flash Flood", InferenceRunner.RenameLabel(label), adjustedProb, useTrust, inputDict);
        return label;
    }
}
