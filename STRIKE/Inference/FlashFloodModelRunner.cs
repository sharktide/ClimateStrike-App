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
