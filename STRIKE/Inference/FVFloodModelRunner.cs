namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class FVFloodModelRunner
{
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🏞️ FLUVIAL FLOOD LIKELY | {prob:F4}",
            < 0.44f => $"🛡️ Fluvial Flood Unlikely | {prob:F4}",
            _      => $"⚠️ Fluvial Flood Possible | {prob:F4}"
        };
    }

    static FVFloodModelRunner()
    {
        var scaler = ScalerLoader.Load("models/scalers/FV-FloodScaler.txt");

        InferenceRunner.Initialize("Models/FV-FloodNet.onnx", "Models/FV-FloodTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        return Eval.GetLabel(adjustedProb);
    }
}
