namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class PVFloodModelRunner
{
    private static class Eval
    {
        public static string GetLabel(float prob) => prob switch
        {
            > 0.49f => $"🌧️ PLUVIAL FLOOD LIKELY | {prob:F4}",
            < 0.44f => $"🛡️ Pluvial Flood Unlikely | {prob:F4}",
            _      => $"⚠️ Pluvial Flood Possible | {prob:F4}"
        };
    }

    static PVFloodModelRunner()
    {
        var scaler = ScalerLoader.Load("Models/scalers/PV-FloodScaler.txt");

        InferenceRunner.Initialize("Models/PV-FloodNet.onnx", "Models/PV-FloodTrustNet.onnx", scaler);
    }

    public static string Predict(float[] inputFeatures, bool useTrust)
    {
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        return Eval.GetLabel(adjustedProb);
    }
}
