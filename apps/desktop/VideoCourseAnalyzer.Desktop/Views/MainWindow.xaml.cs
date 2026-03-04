using System.Windows;
using VideoCourseAnalyzer.Desktop.Services;
using VideoCourseAnalyzer.Desktop.ViewModels;

namespace VideoCourseAnalyzer.Desktop.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = new MainViewModel(new ApiClient());
        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        if (DataContext is MainViewModel vm)
        {
            vm.ChatMessages.CollectionChanged += (_, _) =>
            {
                ChatScrollViewer.ScrollToEnd();
            };
        }
    }
}
