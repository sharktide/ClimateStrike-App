namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class HurricaneModelRunner
{
    public static string[] GetNames()
    {
        return ["Sea Surface Temperature", "Ocean Heat Content", "Mid-Level Humidity", "Vertical Wind Shear", "Potential Vorticity"];
    }
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🌀 HURRICANE LIKELY  | {prob:F4}",
            < 0.44f => $"🛡️ Hurricane Unlikely | {prob:F4}",
            _      => $"⚠️ Hurricane Possible | {prob:F4}"
        };
    }

    static HurricaneModelRunner()
    {
        var scaler = ScalerLoader.Load("Models/scalers/HurricaneScaler.txt");

        InferenceRunner.Initialize("Models/HurricaneNet.onnx", "Models/HurricaneTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var inputDict = GetNames().Zip(inputFeatures, (name, value) => new KeyValuePair<string, float>(name, value))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        string label =  Eval.GetLabel(adjustedProb);
        SysIO.SavePrediction("🌀 Hurricane", InferenceRunner.RenameLabel(label), adjustedProb, useTrust, inputDict);
        return label;
    }
}
