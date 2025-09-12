namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class TornadoModelRunner
{
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
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        return Eval.GetLabel(adjustedProb);
    }
}
