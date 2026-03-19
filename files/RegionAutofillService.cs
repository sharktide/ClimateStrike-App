using Microsoft.JSInterop;
using System.Net.Http.Json;

namespace STRIKE.Services
{
    /// <summary>
    /// Uses browser Geolocation + Open-Meteo / OpenTopoData to derive
    /// region-appropriate default values for each predictor model.
    /// </summary>
    public class RegionAutofillService
    {
        private readonly IJSRuntime _js;
        private readonly HttpClient _http;

        public RegionAutofillService(IJSRuntime js, HttpClient http)
        {
            _js = js;
            _http = http;
        }

        // ── Geolocation ────────────────────────────────────────────────
        public async Task<(double Lat, double Lon)?> GetLocationAsync()
        {
            try
            {
                var coords = await _js.InvokeAsync<double[]>("strikeGetLocation");
                if (coords is { Length: 2 })
                    return (coords[0], coords[1]);
            }
            catch { /* permission denied or unavailable */ }
            return null;
        }

        // ── Open-Meteo: current weather at lat/lon ─────────────────────
        public async Task<WeatherSnapshot?> GetWeatherAsync(double lat, double lon)
        {
            try
            {
                var url = $"https://api.open-meteo.com/v1/forecast" +
                          $"?latitude={lat}&longitude={lon}" +
                          $"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation" +
                          $"&wind_speed_unit=ms";
                var resp = await _http.GetFromJsonAsync<OpenMeteoResponse>(url);
                if (resp?.Current == null) return null;

                return new WeatherSnapshot
                {
                    TemperatureC = resp.Current.Temperature_2m,
                    HumidityPct  = resp.Current.Relative_Humidity_2m,
                    WindSpeedMs  = resp.Current.Wind_Speed_10m,
                    PrecipMmHr   = resp.Current.Precipitation
                };
            }
            catch { return null; }
        }

        // ── OpenTopoData: elevation ────────────────────────────────────
        public async Task<float?> GetElevationAsync(double lat, double lon)
        {
            try
            {
                var url = $"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}";
                var resp = await _http.GetFromJsonAsync<TopoResponse>(url);
                return (float?)resp?.Results?.FirstOrDefault()?.Elevation;
            }
            catch { return null; }
        }

        // ── Build typed defaults ───────────────────────────────────────

        public async Task<EarthquakeDefaults?> GetEarthquakeDefaultsAsync()
        {
            var loc = await GetLocationAsync();
            if (loc == null) return null;
            // Seismic defaults are geography-dependent but cannot be fetched
            // in real-time from a free API; return statistically safe mid-range
            // values that indicate "monitor but not alarming".
            return new EarthquakeDefaults
            {
                MomentRate       = 12.5f,
                DisplacementRate = 30f,
                StressChange     = 250f,
                FocalDepth       = 15f,
                SlipRate         = 5.0f
            };
        }

        public async Task<HurricaneDefaults?> GetHurricaneDefaultsAsync()
        {
            var loc = await GetLocationAsync();
            if (loc == null) return null;
            var wx = await GetWeatherAsync(loc.Value.Lat, loc.Value.Lon);
            return new HurricaneDefaults
            {
                SST      = wx != null ? Math.Clamp(wx.TemperatureC + 2f, 24f, 32f) : 28f,
                OHC      = 80f,
                Humidity = wx != null ? (float)Math.Clamp(wx.HumidityPct, 20, 100) : 65f,
                Shear    = 10f,
                Vorticity = 1.0f
            };
        }

        public async Task<TornadoDefaults?> GetTornadoDefaultsAsync()
        {
            var loc = await GetLocationAsync();
            if (loc == null) return null;
            var wx = await GetWeatherAsync(loc.Value.Lat, loc.Value.Lon);
            return new TornadoDefaults
            {
                SRH   = 220f,
                CAPE  = wx != null && wx.HumidityPct > 70 ? 2500f : 1200f,
                LCL   = 900f,
                Shear = 12f,
                STP   = 1.0f
            };
        }

        public async Task<WildfireDefaults?> GetWildfireDefaultsAsync()
        {
            var loc = await GetLocationAsync();
            if (loc == null) return null;
            var wx  = await GetWeatherAsync(loc.Value.Lat, loc.Value.Lon);
            var elv = await GetElevationAsync(loc.Value.Lat, loc.Value.Lon);
            return new WildfireDefaults
            {
                TemperatureK    = wx != null ? (float)(wx.TemperatureC + 273.15) : 300f,
                Humidity        = wx != null ? (float)Math.Clamp(wx.HumidityPct, 0, 100) : 30f,
                WindSpeedMs     = wx != null ? (float)Math.Clamp(wx.WindSpeedMs, 0, 50) : 5f,
                Elevation       = elv ?? 500f,
                VegetationIndex = 0.5f
            };
        }

        public async Task<FloodDefaults?> GetFloodDefaultsAsync()
        {
            var loc = await GetLocationAsync();
            if (loc == null) return null;
            var wx  = await GetWeatherAsync(loc.Value.Lat, loc.Value.Lon);
            var elv = await GetElevationAsync(loc.Value.Lat, loc.Value.Lon);
            return new FloodDefaults
            {
                RainfallMmHr = wx != null ? (float)Math.Clamp(wx.PrecipMmHr * 1f, 0, 150) : 20f,
                Elevation    = elv ?? 10f
            };
        }
    }

    // ── Value-object defaults ──────────────────────────────────────────
    public record WeatherSnapshot
    {
        public float TemperatureC { get; init; }
        public float HumidityPct  { get; init; }
        public float WindSpeedMs  { get; init; }
        public float PrecipMmHr   { get; init; }
    }
    public record EarthquakeDefaults { public float MomentRate,DisplacementRate,StressChange,FocalDepth,SlipRate; }
    public record HurricaneDefaults  { public float SST,OHC,Humidity,Shear,Vorticity; }
    public record TornadoDefaults    { public float SRH,CAPE,LCL,Shear,STP; }
    public record WildfireDefaults   { public float TemperatureK,Humidity,WindSpeedMs,Elevation,VegetationIndex; }
    public record FloodDefaults      { public float RainfallMmHr, Elevation; }

    // ── Open-Meteo DTOs ───────────────────────────────────────────────
    internal class OpenMeteoResponse
    {
        public OpenMeteoCurrent? Current { get; set; }
    }
    internal class OpenMeteoCurrent
    {
        public float Temperature_2m        { get; set; }
        public float Relative_Humidity_2m  { get; set; }
        public float Wind_Speed_10m        { get; set; }
        public float Precipitation         { get; set; }
    }
    internal class TopoResponse
    {
        public List<TopoResult>? Results { get; set; }
    }
    internal class TopoResult
    {
        public float Elevation { get; set; }
    }
}
