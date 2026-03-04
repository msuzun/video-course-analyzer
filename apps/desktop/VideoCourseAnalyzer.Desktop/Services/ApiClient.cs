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

    private static double? GetDoubleOrNull(JsonNode? node)
    {
        if (node is null) return null;
        try
        {
            if (node is JsonValue jv && jv.TryGetValue(out double d))
                return d;
            return node.GetValue<double>();
        }
        catch { /* ignore */ }
        return null;
    }

    public sealed class ChatResult
    {
        public string Answer { get; set; } = string.Empty;
        public List<ChatSource> Sources { get; set; } = [];
    }

    public sealed class ChatSource
    {
        public string ChunkId { get; set; } = string.Empty;
        public string Snippet { get; set; } = string.Empty;
        public double? T0 { get; set; }
        public double? T1 { get; set; }
    }

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

    public async Task<string> CreateChatSessionAsync(string jobId, CancellationToken cancellationToken = default)
    {
        using var response = await HttpClient.PostAsync($"/jobs/{jobId}/chat/sessions", content: null, cancellationToken);
        response.EnsureSuccessStatusCode();

        var raw = await response.Content.ReadAsStringAsync(cancellationToken);
        var doc = JsonNode.Parse(raw);
        var sessionId = doc?["session_id"]?.GetValue<string>();
        if (string.IsNullOrWhiteSpace(sessionId))
        {
            throw new InvalidOperationException("API response does not include session_id.");
        }

        return sessionId;
    }

    public async Task<ChatResult> SendChatAsync(
        string jobId,
        string sessionId,
        string message,
        int topK,
        CancellationToken cancellationToken = default)
    {
        var payload = new
        {
            session_id = sessionId,
            message,
            top_k = topK,
        };
        var json = JsonSerializer.Serialize(payload);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using var response = await HttpClient.PostAsync($"/jobs/{jobId}/chat", content, cancellationToken);
        response.EnsureSuccessStatusCode();

        var raw = await response.Content.ReadAsStringAsync(cancellationToken);
        var node = JsonNode.Parse(raw) as JsonObject;
        if (node is null)
        {
            throw new InvalidOperationException("API response is invalid.");
        }

        var result = new ChatResult
        {
            Answer = node["answer"]?.GetValue<string>() ?? string.Empty,
            Sources = [],
        };

        var sources = node["sources"] as JsonArray;
        if (sources is not null)
        {
            foreach (var item in sources.OfType<JsonObject>())
            {
                result.Sources.Add(
                    new ChatSource
                    {
                        ChunkId = item["chunk_id"]?.GetValue<string>() ?? string.Empty,
                        Snippet = item["snippet"]?.GetValue<string>() ?? string.Empty,
                        T0 = GetDoubleOrNull(item["t0"]),
                        T1 = GetDoubleOrNull(item["t1"]),
                    });
            }
        }

        return result;
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
