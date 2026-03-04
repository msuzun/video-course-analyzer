namespace VideoCourseAnalyzer.Desktop.Models;

public sealed class SourceItem
{
    public string ChunkId { get; set; } = string.Empty;
    public string Snippet { get; set; } = string.Empty;
    public double? T0 { get; set; }
    public double? T1 { get; set; }

    /// <summary>Formatted for display and copy, e.g. "02:12 - 03:00".</summary>
    public string TimestampDisplay
    {
        get
        {
            if (!T0.HasValue) return "0:00 - 0:00";
            var s0 = (int)Math.Floor(T0.Value);
            var m0 = s0 / 60;
            var sec0 = s0 % 60;
            if (!T1.HasValue) return $"{m0:D2}:{sec0:D2} - 0:00";
            var s1 = (int)Math.Floor(T1.Value);
            var m1 = s1 / 60;
            var sec1 = s1 % 60;
            return $"{m0:D2}:{sec0:D2} - {m1:D2}:{sec1:D2}";
        }
    }
}
