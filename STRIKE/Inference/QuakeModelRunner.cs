namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class QuakeModelRunner
{
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🌎 EARTHQUAKE LIKELY | {prob:F4}",
            < 0.44f => $"🛡️ Earthquake Unlikely | {prob:F4}",
            _      => $"⚠️ Earthquake Possible | {prob:F4}"
        };
    }

    static QuakeModelRunner()
    {
        var scaler = ScalerLoader.Load("Models/scalers/QuakeScaler.txt");

        InferenceRunner.Initialize("Models/QuakeNet.onnx", "Models/QuakeTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        return Eval.GetLabel(adjustedProb);
    }
}
