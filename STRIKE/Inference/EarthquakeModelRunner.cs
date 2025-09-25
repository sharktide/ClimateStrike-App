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

    static EarthquakeModelRunner()
    {
        var scaler = ScalerLoader.Load("Models/scalers/QuakeScaler.txt");

        InferenceRunner.Initialize("Models/QuakeNet.onnx", "Models/QuakeTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var inputDict = GetNames().Zip(inputFeatures, (name, value) => new KeyValuePair<string, float>(name, value))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        string label = Eval.GetLabel(adjustedProb);
        SysIO.SavePrediction("🌎 Earthquake", InferenceRunner.RenameLabel(label), adjustedProb, useTrust, inputDict);
        return label;
    }
}
