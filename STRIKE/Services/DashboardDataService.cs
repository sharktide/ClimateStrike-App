using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using STRIKE.InferenceUtils;


namespace STRIKE.Services
{
    public class RowData
    {
        public string Name { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
    }
    public class InputDisplayRow
    {
        public string Name { get; set; } = string.Empty;
        public float Value { get; set; }
        public string Unit { get; set; } = string.Empty;
    }

    public class PredictionResult
    {
        public required string DisasterType { get; set; } = string.Empty;
        public required string Label { get; set; } = string.Empty;
        public required float Probability { get; set; }
        public required bool Trust { get; set; }
        public required Dictionary<string, float> Inputs { get; set; }
        public required DateTime Timestamp { get; set; }
    }


    public interface IDashboardDataService
    {
        Task<List<PredictionResult>> GetRecentPredictionsAsync();
        Task<Dictionary<string, float>> GetModelConfidencesAsync();
        Task DeletePredictionAsync(DateTime timestamp);
        Task DeleteAllPredictionsAsync();
    }

    public class MockDashboardDataService : IDashboardDataService
    {
        public Task<List<PredictionResult>> GetRecentPredictionsAsync()
        {
            var history = SysIO.GetPredictionHistory();
            return Task.FromResult(history);
        }

        public Task<Dictionary<string, float>> GetModelConfidencesAsync()
        {
            return Task.FromResult(new Dictionary<string, float>
            {
                { "Wildfires", 0.98f },
                { "Fluvial Floods", 0.96f },
                { "Pluvial Floods", 0.95f },
                { "Flash Floods", 0.97f },
                { "Earthquakes", 0.95f },
                { "Hurricanes", 0.99f },
                { "Tornadoes", 0.95f}
            });
        }

        public Task DeletePredictionAsync(DateTime timestamp)
        {
            SysIO.DeletePrediction(timestamp);
            return Task.CompletedTask;
        }

        public Task DeleteAllPredictionsAsync()
        {
            SysIO.DeleteAllPredictions();
            return Task.CompletedTask;
        }
    }
}
