namespace VideoCourseAnalyzer.Desktop.Models;

public sealed class SourceItem
{
    public string ChunkId { get; set; } = string.Empty;
    public string Snippet { get; set; } = string.Empty;
    public double? T0 { get; set; }
    public double? T1 { get; set; }
}
