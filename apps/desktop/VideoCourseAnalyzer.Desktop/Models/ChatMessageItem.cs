namespace VideoCourseAnalyzer.Desktop.Models;

public sealed class ChatMessageItem
{
    public string Role { get; set; } = "assistant";
    public string Content { get; set; } = string.Empty;
}
