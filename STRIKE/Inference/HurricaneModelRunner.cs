namespace STRIKE.Inference;
using STRIKE.InferenceUtils;

public static class HurricaneModelRunner
{
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
        var adjustedProb = InferenceRunner.RunPrediction(inputFeatures, useTrust);
        return Eval.GetLabel(adjustedProb);
    }
}
