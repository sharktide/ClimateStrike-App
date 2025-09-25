namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class WildfireModelRunner
{
    public static string[] GetNames()
    {
        return ["Temperature", "Humidity", "Wind Speed", "Elevation", "Vegitation Index"];
    }
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🔥 FIRE LIKELY | {prob:F4}",
            < 0.44f => $"🛡️ Fire Unlikely | {prob:F4}",
            _      => $"⚠️ Fire Possible | {prob:F4}"
        };
    }

    static WildfireModelRunner()
    {
        var scaler = ScalerLoader.Load("Models/scalers/FireScaler.txt");

        InferenceRunner.Initialize("Models/FireNet.onnx", "Models/FireTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var inputDict = GetNames().Zip(inputFeatures, (name, value) => new KeyValuePair<string, float>(name, value))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        string label =  Eval.GetLabel(adjustedProb);
        SysIO.SavePrediction("🔥 Wildfire", InferenceRunner.RenameLabel(label), adjustedProb, useTrust, inputDict);
        return label;
    }
}
