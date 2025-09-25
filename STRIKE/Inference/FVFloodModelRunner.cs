namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class FVFloodModelRunner
{
    public static string[] GetNames()
    {
        return ["Rainfall Intensity", "Relative Water Level", "Relative Slope", "Relative Elevation", "Distance From River"];
    }
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🏞️ FLUVIAL FLOOD LIKELY | {prob:F4}",
            < 0.44f => $"🛡️ Fluvial Flood Unlikely | {prob:F4}",
            _ => $"⚠️ Fluvial Flood Possible | {prob:F4}"
        };
    }

    static FVFloodModelRunner()
    {
        var scaler = ScalerLoader.Load("Models/scalers/FV-FloodScaler.txt");

        InferenceRunner.Initialize("Models/FV-FloodNet.onnx", "Models/FV-FloodTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var inputDict = GetNames().Zip(inputFeatures, (name, value) => new KeyValuePair<string, float>(name, value))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        string label =  Eval.GetLabel(adjustedProb);
        SysIO.SavePrediction("🏞️ Fluvial Flood", InferenceRunner.RenameLabel(label), adjustedProb, useTrust, inputDict);
        return label;
    }
}
