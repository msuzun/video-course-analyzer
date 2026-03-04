using System.Collections.ObjectModel;
using System.Collections.Generic;
using System.Text.Json.Nodes;
using System.Windows;
using VideoCourseAnalyzer.Desktop.Models;
using VideoCourseAnalyzer.Desktop.Services;

namespace VideoCourseAnalyzer.Desktop.ViewModels;

public sealed class MainViewModel : ViewModelBase
{
    private readonly AsyncRelayCommand _startAnalysisCommand;
    private readonly HashSet<string> _briefLoadedJobs = [];
    private CancellationTokenSource? _eventCts;
    private bool _isStarting;
    private string _newAnalysisUrl = string.Empty;
    private string _currentStatus = "IDLE";
    private string _currentStep = "-";
    private double _progressValue;
    private JobItem? _selectedJob;

    public MainViewModel(ApiClient apiClient)
    {
        ApiClient = apiClient;
        _startAnalysisCommand = new AsyncRelayCommand(StartAnalysisAsync, CanStartAnalysis);
    }

    public ApiClient ApiClient { get; }

    public AsyncRelayCommand StartAnalysisCommand => _startAnalysisCommand;

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
        }
    }

    private bool CanStartAnalysis()
    {
        return !IsStarting && !string.IsNullOrWhiteSpace(NewAnalysisUrl);
    }

    private async Task StartAnalysisAsync()
    {
        IsStarting = true;
        try
        {
            var jobId = await ApiClient.CreateJobAsync(NewAnalysisUrl.Trim());

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
                NewAnalysisUrl = string.Empty;
            });

            _eventCts?.Cancel();
            _eventCts = new CancellationTokenSource();
            _ = ListenEventsAsync(jobId, _eventCts.Token);
        }
        finally
        {
            IsStarting = false;
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
            RaisePropertyChanged(nameof(SelectedJob));
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
            briefMarkdown = "Video hazir ✅\n\nVideo brief bulunamadi.";
        }

        await Application.Current.Dispatcher.InvokeAsync(() =>
        {
            Sources.Clear();
            ChatMessages.Add(
                new MessageItem
                {
                    Role = MessageRole.System,
                    Text = $"Video hazir ✅\n\n{briefMarkdown}",
                    Timestamp = DateTime.Now,
                });
        });
    }
}
