namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class FlashFloodModelRunner
{
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
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        return Eval.GetLabel(adjustedProb);
    }
}
