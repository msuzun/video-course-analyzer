namespace VideoCourseAnalyzer.Desktop.Models;

public sealed class StepItem
{
    public string Name { get; set; } = string.Empty;
    public string State { get; set; } = "UNKNOWN";
    public double Progress { get; set; }
}
