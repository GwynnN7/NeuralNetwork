#include "cli.hpp"

#include "CLI/CLI.hpp"
#include "types.hpp"

#include <print>

Args Args::parse(int argc, char* argv[]) {
    Args args;
    CLI::App app{"Neural Network Training"};

    // Epochs parameters
    app.add_option("--updates", args.updates, "Number of weight updates per run")->default_val(800)->check(CLI::PositiveNumber);
    app.add_option("--patience", args.patience, "Weight updates without improvement before early stopping (0 = disabled)")->default_val(75)->check(CLI::NonNegativeNumber);
    app.add_option("--warmup", args.warmup, "Weight updates spent increasing the learning rate (0 = disabled)")->default_val(50)->check(CLI::NonNegativeNumber);
    app.add_option("--stopping", args.stopping_rule, "Early stopping rule")->transform(CLI::CheckedTransformer(Lookup::str_to_stopping, CLI::ignore_case))->default_val(StoppingRule::PATIENCE);

    // Dataset parameters
    app.add_option("dataset", args.dataset_type, "Dataset type")->transform(CLI::CheckedTransformer(Lookup::str_to_dataset, CLI::ignore_case))->required();
    app.add_option("--train_ratio", args.train_ratio, "Training set ratio, exclusive bounds")->default_val(0.85)->check(CLI::Range(0.1, 1.0).description("in (0,1)"));
    app.add_option("--dataset_ratio", args.dataset_ratio, "Subset of dataset used (when applicable)")->default_val(1.0)->check(CLI::Range(0.1, 1.0).description("in (0,1]"));
    app.add_option("--normalization", args.normalization_type, "Normalization type")->transform(CLI::CheckedTransformer(Lookup::str_to_normalization, CLI::ignore_case))->default_val(NormalizationType::NONE);

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