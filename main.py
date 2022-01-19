from endofoutbreak.pipelines.backtest import run_backtest_pipeline


if __name__ == "__main__":
    run_backtest_pipeline(
        "taiwan",
        plot_model_distributions=True,
        rerun_mcmc=True,
        generate_trace_plots=True,
        generate_eoo_plot=True,
    )
