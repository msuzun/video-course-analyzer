using System.Net.Http;
using System.Net.Http.Headers;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace VideoCourseAnalyzer.Desktop.Services;

public sealed class ApiClient
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public ApiClient()
    {
        HttpClient = new HttpClient
        {
            BaseAddress = new Uri("http://localhost:8000"),
        };
    }

    public HttpClient HttpClient { get; }

    public async Task<string> CreateJobAsync(string sourceUrl, CancellationToken cancellationToken = default)
    {
        var requestPayload = new
        {
            source_type = "video_url",
            source_url = sourceUrl,
            options = new { },
        };

        var json = JsonSerializer.Serialize(requestPayload);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using var response = await HttpClient.PostAsync("/jobs", content, cancellationToken);
        response.EnsureSuccessStatusCode();

        var raw = await response.Content.ReadAsStringAsync(cancellationToken);
        var doc = JsonNode.Parse(raw);
        var jobId = doc?["job_id"]?.GetValue<string>();
        if (string.IsNullOrWhiteSpace(jobId))
        {
            throw new InvalidOperationException("API response does not include job_id.");
        }

        return jobId;
    }

    public async Task<string> GetArtifactTextAsync(
        string jobId,
        string artifactKey,
        CancellationToken cancellationToken = default)
    {
        using var response = await HttpClient.GetAsync($"/jobs/{jobId}/artifacts/{artifactKey}", cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync(cancellationToken);
    }

    public async Task ListenJobEventsAsync(
        string jobId,
        Func<string, JsonObject, Task> onEvent,
        CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, $"/jobs/{jobId}/events");
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

        using var response = await HttpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        string? currentEvent = null;
        var dataBuffer = new StringBuilder();

        while (!cancellationToken.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(cancellationToken);
            if (line is null)
            {
                break;
            }

            if (line.Length == 0)
            {
                if (!string.IsNullOrWhiteSpace(currentEvent) && dataBuffer.Length > 0)
                {
                    var rawData = dataBuffer.ToString().Trim();
                    var node = JsonNode.Parse(rawData) as JsonObject;
                    if (node is not null)
                    {
                        await onEvent(currentEvent, node);
                    }
                }

                currentEvent = null;
                dataBuffer.Clear();
                continue;
            }

            if (line.StartsWith("event:", StringComparison.Ordinal))
            {
                currentEvent = line["event:".Length..].Trim();
                continue;
            }

            if (line.StartsWith("data:", StringComparison.Ordinal))
            {
                dataBuffer.AppendLine(line["data:".Length..].Trim());
            }
        }
    }
}
