using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
namespace TrainDosy;

public sealed record FitRequest(
    [property: JsonPropertyName("Y")] double[][] Y,
    [property: JsonPropertyName("b")] double[] B,
    [property: JsonPropertyName("ppm")] double[] Ppm,
    [property: JsonPropertyName("sigma")] double Sigma,
    [property: JsonPropertyName("method")] string Method = "RAI-S",
    [property: JsonPropertyName("mask")] bool[]? Mask = null,
    [property: JsonPropertyName("diffusion_bounds")] double[]? DiffusionBounds = null,
    [property: JsonPropertyName("bins")] int Bins = 256,
    [property: JsonPropertyName("max_components")] int MaxComponents = 4);

public sealed record FitResult {
    [JsonPropertyName("method")] public required string Method {get;init;}
    [JsonPropertyName("selected_rank")] public int SelectedRank {get;init;}
    [JsonPropertyName("success")] public bool Success {get;init;}
    [JsonPropertyName("kkt")] public double? Kkt {get;init;}
    [JsonPropertyName("D")] public double[]? Rates {get;init;}
    [JsonPropertyName("D_grid")] public double[]? DiffusionGrid {get;init;}
    [JsonPropertyName("A")] public required double[][] A {get;init;}
    [JsonPropertyName("X")] public required double[][] X {get;init;}
    [JsonPropertyName("prediction")] public required double[][] Prediction {get;init;}
    [JsonPropertyName("mask")] public required bool[] Mask {get;init;}
    [JsonExtensionData] public Dictionary<string,JsonElement>? Diagnostics {get;init;}
}

/// <summary>Typed, cancellable client. Numerical inference runs in the reference backend.</summary>
public sealed class DosyClient(HttpClient http) {
    public static JsonSerializerOptions JsonOptions {get;} = new() {
        PropertyNameCaseInsensitive=false, DefaultIgnoreCondition=JsonIgnoreCondition.WhenWritingNull,
        WriteIndented=true
    };
    public async Task<FitResult> FitAsync(FitRequest request,CancellationToken cancellationToken=default) {
        using var response=await http.PostAsJsonAsync("v1/fit",request,JsonOptions,cancellationToken);
        if(!response.IsSuccessStatusCode) {
            var body=await response.Content.ReadAsStringAsync(cancellationToken);
            throw new HttpRequestException($"DOSY {(int)response.StatusCode}: {body}",null,response.StatusCode);
        }
        return await response.Content.ReadFromJsonAsync<FitResult>(JsonOptions,cancellationToken)
            ?? throw new InvalidDataException("Empty DOSY response");
    }
}
