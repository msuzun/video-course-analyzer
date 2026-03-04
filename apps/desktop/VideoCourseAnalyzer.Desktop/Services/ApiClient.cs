using System.Net.Http;

namespace VideoCourseAnalyzer.Desktop.Services;

public sealed class ApiClient
{
    public ApiClient()
    {
        HttpClient = new HttpClient
        {
            BaseAddress = new Uri("http://localhost:8000"),
        };
    }

    public HttpClient HttpClient { get; }
}
