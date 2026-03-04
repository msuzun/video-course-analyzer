using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text.Json.Nodes;
using System.Windows;
using VideoCourseAnalyzer.Desktop.Models;
using VideoCourseAnalyzer.Desktop.Services;

namespace VideoCourseAnalyzer.Desktop.ViewModels;

public sealed class MainViewModel : ViewModelBase
{
    private readonly AsyncRelayCommand _startAnalysisCommand;
    private readonly AsyncRelayCommand _sendChatCommand;
    private readonly RelayCommand _copyTimestampCommand;
    private readonly HashSet<string> _briefLoadedJobs = [];
    private readonly Dictionary<string, string> _chatSessionByJob = [];

    private CancellationTokenSource? _eventCts;
    private bool _isStarting;
    private bool _isSendingChat;
    private string _newAnalysisUrl = string.Empty;
    private string _chatInput = string.Empty;
    private string _currentStatus = "IDLE";
    private string _currentStep = "-";
    private double _progressValue;
    private JobItem? _selectedJob;

    public MainViewModel(ApiClient apiClient)
    {
        ApiClient = apiClient;
        _startAnalysisCommand = new AsyncRelayCommand(StartAnalysisAsync, CanStartAnalysis);
        _sendChatCommand = new AsyncRelayCommand(SendChatAsync, CanSendChat);
        _copyTimestampCommand = new RelayCommand(CopyTimestamp, CanCopyTimestamp);
    }

    public ApiClient ApiClient { get; }

    public AsyncRelayCommand StartAnalysisCommand => _startAnalysisCommand;
    public AsyncRelayCommand SendChatCommand => _sendChatCommand;
    public RelayCommand CopyTimestampCommand => _copyTimestampCommand;

    public ObservableCollection<JobItem> Jobs { get; } = [];
    public ObservableCollection<MessageItem> ChatMessages { get; } = [];
    public ObservableCollection<SourceItem> Sources { get; } = [];
    public ObservableCollection<StepItem> StepTimeline { get; } = [];
    public ObservableCollection<string> LiveLogs { get; } = [];

    public string NewAnalysisUrl
    {
        get => _newAnalysisUrl;
        set
        {
            if (_newAnalysisUrl == value)
            {
                return;
            }

            _newAnalysisUrl = value;
            RaisePropertyChanged();
            _startAnalysisCommand.RaiseCanExecuteChanged();
        }
    }

    public string ChatInput
    {
        get => _chatInput;
        set
        {
            if (_chatInput == value)
            {
                return;
            }

            _chatInput = value;
            RaisePropertyChanged();
            _sendChatCommand.RaiseCanExecuteChanged();
        }
    }

    public bool IsStarting
    {
        get => _isStarting;
        private set
        {
            if (_isStarting == value)
            {
                return;
            }

            _isStarting = value;
            RaisePropertyChanged();
            _startAnalysisCommand.RaiseCanExecuteChanged();
        }
    }

    public bool IsSendingChat
    {
        get => _isSendingChat;
        private set
        {
            if (_isSendingChat == value)
            {
                return;
            }

            _isSendingChat = value;
            RaisePropertyChanged();
            _sendChatCommand.RaiseCanExecuteChanged();
        }
    }

    public string CurrentStatus
    {
        get => _currentStatus;
        private set
        {
            if (_currentStatus == value)
            {
                return;
            }

            _currentStatus = value;
            RaisePropertyChanged();
        }
    }

    public string CurrentStep
    {
        get => _currentStep;
        private set
        {
            if (_currentStep == value)
            {
                return;
            }

            _currentStep = value;
            RaisePropertyChanged();
        }
    }

    public double ProgressValue
    {
        get => _progressValue;
        private set
        {
            if (Math.Abs(_progressValue - value) < 0.0001)
            {
                return;
            }

            _progressValue = value;
            RaisePropertyChanged();
        }
    }

    public JobItem? SelectedJob
    {
        get => _selectedJob;
        set
        {
            if (_selectedJob == value)
            {
                return;
            }

            _selectedJob = value;
            RaisePropertyChanged();
            _sendChatCommand.RaiseCanExecuteChanged();
        }
    }

    private bool CanStartAnalysis() => !IsStarting && !string.IsNullOrWhiteSpace(NewAnalysisUrl);

    private bool CanSendChat() => !IsSendingChat && SelectedJob is not null && !string.IsNullOrWhiteSpace(ChatInput);

    private async Task StartAnalysisAsync()
    {
        var sourceUrl = NewAnalysisUrl.Trim();
        if (!Uri.TryCreate(sourceUrl, UriKind.Absolute, out var uri) ||
            (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
        {
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                LiveLogs.Add("[client] start failed: invalid url");
                ChatMessages.Add(
                    new MessageItem
                    {
                        Role = MessageRole.System,
                        Text = "Gecerli bir URL girin (http/https).",
                        Timestamp = DateTime.Now,
                    });
            });
            return;
        }

        IsStarting = true;
        try
        {
            var jobId = await ApiClient.CreateJobAsync(sourceUrl);

            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                var item = new JobItem
                {
                    JobId = jobId,
                    Status = "QUEUED",
                };
                Jobs.Insert(0, item);
                SelectedJob = item;
                CurrentStatus = "QUEUED";
                CurrentStep = "-";
                ProgressValue = 0;
                StepTimeline.Clear();
                LiveLogs.Clear();
                ChatMessages.Clear();
                Sources.Clear();
                NewAnalysisUrl = string.Empty;
                ChatInput = string.Empty;
            });

            _eventCts?.Cancel();
            _eventCts = new CancellationTokenSource();
            _ = ListenEventsAsync(jobId, _eventCts.Token);
        }
        catch (Exception ex)
        {
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                LiveLogs.Add($"[client] start failed: {ex.Message}");
                ChatMessages.Add(
                    new MessageItem
                    {
                        Role = MessageRole.System,
                        Text = $"Analiz baslatilamadi: {ex.Message}",
                        Timestamp = DateTime.Now,
                    });
            });
        }
        finally
        {
            IsStarting = false;
        }
    }

    private async Task SendChatAsync()
    {
        var selected = SelectedJob;
        if (selected is null)
        {
            return;
        }

        var question = ChatInput.Trim();
        if (string.IsNullOrWhiteSpace(question))
        {
            return;
        }

        IsSendingChat = true;
        ChatInput = string.Empty;

        try
        {
            var jobId = selected.JobId;
            if (!_chatSessionByJob.TryGetValue(jobId, out var sessionId) || string.IsNullOrWhiteSpace(sessionId))
            {
                sessionId = await ApiClient.CreateChatSessionAsync(jobId);
                _chatSessionByJob[jobId] = sessionId;
            }

            ChatMessages.Add(
                new MessageItem
                {
                    Role = MessageRole.User,
                    Text = question,
                    Timestamp = DateTime.Now,
                });

            var chatResult = await ApiClient.SendChatAsync(jobId, sessionId, question, topK: 6);

            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                ChatMessages.Add(
                    new MessageItem
                    {
                        Role = MessageRole.Assistant,
                        Text = string.IsNullOrWhiteSpace(chatResult.Answer)
                            ? "This topic is not covered in the video."
                            : chatResult.Answer,
                        Timestamp = DateTime.Now,
                    });

                Sources.Clear();
                foreach (var source in chatResult.Sources)
                {
                    Sources.Add(
                        new SourceItem
                        {
                            ChunkId = source.ChunkId ?? string.Empty,
                            Snippet = source.Snippet ?? string.Empty,
                            T0 = source.T0,
                            T1 = source.T1,
                        });
                }
            });
        }
        catch (Exception ex)
        {
            ChatMessages.Add(
                new MessageItem
                {
                    Role = MessageRole.System,
                    Text = $"Chat error: {ex.Message}",
                    Timestamp = DateTime.Now,
                });
        }
        finally
        {
            IsSendingChat = false;
        }
    }

    private async Task ListenEventsAsync(string jobId, CancellationToken cancellationToken)
    {
        try
        {
            await ApiClient.ListenJobEventsAsync(
                jobId,
                async (eventType, data) =>
                {
                    await Application.Current.Dispatcher.InvokeAsync(() =>
                    {
                        if (eventType == "state")
                        {
                            ApplyStateEvent(jobId, data);
                            return;
                        }

                        if (eventType == "log")
                        {
                            ApplyLogEvent(data);
                        }
                    });
                },
                cancellationToken);
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                LiveLogs.Add($"[client] event stream stopped: {ex.Message}");
            });
        }
    }

    private void ApplyStateEvent(string jobId, JsonObject data)
    {
        var payload = data["payload"] as JsonObject;
        if (payload is null)
        {
            return;
        }

        CurrentStatus = payload["status"]?.GetValue<string>() ?? "UNKNOWN";
        CurrentStep = payload["current_step"]?.GetValue<string>() ?? "-";
        ProgressValue = payload["progress"]?.GetValue<double>() ?? 0.0;

        if (SelectedJob is not null && SelectedJob.JobId == jobId)
        {
            SelectedJob.Status = CurrentStatus;
        }

        StepTimeline.Clear();
        var steps = payload["steps"] as JsonArray;
        if (steps is not null)
        {
            foreach (var step in steps.OfType<JsonObject>())
            {
                StepTimeline.Add(
                    new StepItem
                    {
                        Name = step["name"]?.GetValue<string>() ?? "-",
                        State = step["state"]?.GetValue<string>() ?? "UNKNOWN",
                        Progress = step["progress"]?.GetValue<double>() ?? 0.0,
                    });
            }
        }

        if ((string.Equals(CurrentStatus, "DONE", StringComparison.OrdinalIgnoreCase) ||
             string.Equals(CurrentStatus, "COMPLETED", StringComparison.OrdinalIgnoreCase)) &&
            _briefLoadedJobs.Add(jobId))
        {
            _ = LoadBriefMessageAsync(jobId);
        }
    }

    private void ApplyLogEvent(JsonObject data)
    {
        var payload = data["payload"] as JsonObject;
        if (payload is null)
        {
            return;
        }

        var line = payload["line"]?.GetValue<string>();
        if (string.IsNullOrWhiteSpace(line))
        {
            return;
        }

        LiveLogs.Add(line);
        while (LiveLogs.Count > 500)
        {
            LiveLogs.RemoveAt(0);
        }
    }

    private async Task LoadBriefMessageAsync(string jobId)
    {
        string briefMarkdown;
        try
        {
            briefMarkdown = await ApiClient.GetArtifactTextAsync(jobId, "video_brief_md");
        }
        catch
        {
            briefMarkdown = "Video brief could not be loaded.";
        }

        await Application.Current.Dispatcher.InvokeAsync(() =>
        {
            Sources.Clear();
            ChatMessages.Add(
                new MessageItem
                {
                    Role = MessageRole.System,
                    Text = $"Video analysis complete. Here is the video brief.\n\n{briefMarkdown}",
                    Timestamp = DateTime.Now,
                });
        });
    }

    private bool CanCopyTimestamp(object? parameter)
    {
        return parameter is SourceItem source && source.T0.HasValue;
    }

    private void CopyTimestamp(object? parameter)
    {
        if (parameter is not SourceItem source || !source.T0.HasValue)
        {
            return;
        }

        Clipboard.SetText(source.TimestampDisplay);
    }
}
