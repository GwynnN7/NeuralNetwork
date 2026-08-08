#include "cli.hpp"

#include "CLI/CLI.hpp"
#include "types.hpp"

#include <print>

Args Args::parse(int argc, char* argv[]) {
    Args args;
    CLI::App app{"Neural Network Training"};

    // Epochs parameters
    app.add_option("--epochs", args.epochs, "Number of epochs")->default_val(800)->check(CLI::PositiveNumber);
    app.add_option("--patience", args.patience, "Patience for early stopping, as a fraction of epochs (0 = disabled)")->default_val(0.125)->check(CLI::Range(0.0, 0.5));
    app.add_option("--warmup", args.warmup, "Learning rate warmup, as a fraction of epochs (0 = disabled)")->default_val(0.1)->check(CLI::Range(0.0, 0.5));

    // Dataset parameters
    app.add_option("dataset", args.dataset_type, "Dataset type")->transform(CLI::CheckedTransformer(Maps::str_to_dataset, CLI::ignore_case))->required();
    app.add_option("--train_ratio", args.train_ratio, "Training set ratio, exclusive bounds")->default_val(0.85)->check(CLI::Range(0.0, 1.0).description("in (0,1)"));
    app.add_option("--dataset_ratio", args.dataset_ratio, "Subset of dataset used (when applicable)")->default_val(1.0)->check(CLI::Range(0.0, 1.0).description("in (0,1]"));

    // Cross-validation parameters
    app.add_option("--grid", args.grid_file, "File containing model parameters for grid search")->default_val("grids/grid.csv");
    app.add_option("--trials", args.trials, "Random restarts per training run")->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("--inner-k", args.inner_folds, "Number of folds for inner cross-validation")->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("--outer-k", args.outer_folds, "Number of folds for outer cross-validation")->default_val(1)->check(CLI::PositiveNumber);
    app.add_flag("--shuffle", args.shuffle, "Shuffle dataset before splitting into folds");

    // Configuration parameters
    app.add_option("--name", args.name, "Name for the model and log files")->default_val("model");
    app.add_option("--seed", args.seed, "Random seed")->default_val(42);
    app.add_flag("--dump", args.dump, "Dump best models' weights to file after total retraining")->default_val(false);
    app.add_flag("--train", args.train, "Train a new model")->default_val(false);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::println(stderr, "Error parsing arguments: {}", e.what());
        throw std::invalid_argument("Invalid command line arguments");
    }

    return args;
}