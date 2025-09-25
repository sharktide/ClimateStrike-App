namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class TornadoModelRunner
{
    public static string[] GetNames()
    {
        return ["Storm Relative Helicity", "CAPE", "Lifted Condensation Level", "Bulk Wind Shear", "Significant Tornado Parameter"];
    }
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🌪️ TORNADO LIKELY | {prob:F4}",
            < 0.44f => $"🛡️ Tornado Unlikely | {prob:F4}",
            _      => $"⚠️ Tornado Possible | {prob:F4}"
        };
    }

    static TornadoModelRunner()
    {
        var scaler = ScalerLoader.Load("Models/scalers/TornadoScaler.txt");

        InferenceRunner.Initialize("Models/TornadoNet.onnx", "Models/TornadoTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var inputDict = GetNames().Zip(inputFeatures, (name, value) => new KeyValuePair<string, float>(name, value))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        string label =  Eval.GetLabel(adjustedProb);
        SysIO.SavePrediction("🌪️ Tornado", InferenceRunner.RenameLabel(label), adjustedProb, useTrust, inputDict);
        return label;
    }
}
