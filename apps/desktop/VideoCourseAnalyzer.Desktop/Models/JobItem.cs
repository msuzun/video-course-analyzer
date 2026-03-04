namespace VideoCourseAnalyzer.Desktop.Models;

public sealed class JobItem
{
    public string JobId { get; set; } = string.Empty;
    public string Status { get; set; } = "QUEUED";
}
