namespace VideoCourseAnalyzer.Desktop.Models;

public enum MessageRole
{
    System,
    User,
    Assistant,
}

public sealed class MessageItem
{
    public MessageRole Role { get; set; } = MessageRole.System;
    public string Text { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.Now;
}
