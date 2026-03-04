using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace VideoCourseAnalyzer.Desktop.Models;

public sealed class JobItem : INotifyPropertyChanged
{
    private string _jobId = string.Empty;
    private string _status = "QUEUED";

    public event PropertyChangedEventHandler? PropertyChanged;

    public string JobId
    {
        get => _jobId;
        set
        {
            if (_jobId == value)
            {
                return;
            }

            _jobId = value;
            RaisePropertyChanged();
        }
    }

    public string Status
    {
        get => _status;
        set
        {
            if (_status == value)
            {
                return;
            }

            _status = value;
            RaisePropertyChanged();
        }
    }

    private void RaisePropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
