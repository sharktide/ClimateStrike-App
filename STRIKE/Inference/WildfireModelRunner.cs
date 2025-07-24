namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class WildfireModelRunner
{
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
        var scaler = ScalerLoader.Load("models/scalers/FireScaler.txt");

        InferenceRunner.Initialize("Models/FireNet.onnx", "Models/FireTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        return Eval.GetLabel(adjustedProb);
    }
}
