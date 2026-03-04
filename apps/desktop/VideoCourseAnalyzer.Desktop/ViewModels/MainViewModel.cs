using System.Collections.ObjectModel;
using VideoCourseAnalyzer.Desktop.Models;
using VideoCourseAnalyzer.Desktop.Services;

namespace VideoCourseAnalyzer.Desktop.ViewModels;

public sealed class MainViewModel : ViewModelBase
{
    private JobItem? _selectedJob;

    public MainViewModel(ApiClient apiClient)
    {
        ApiClient = apiClient;
    }

    public ApiClient ApiClient { get; }

    public ObservableCollection<JobItem> Jobs { get; } = [];

    public ObservableCollection<ChatMessageItem> ChatMessages { get; } = [];

    public ObservableCollection<SourceItem> Sources { get; } = [];

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
}
