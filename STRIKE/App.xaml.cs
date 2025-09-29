namespace STRIKE
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();
        }

        public static string GetVersion()
        {
            return "1.0.0";
        }

        protected override Window CreateWindow(IActivationState? activationState)
        {
            return new Window(new MainPage()) { Title = "ClimateStrike AI v1.0.0" };
        }
    }
}
