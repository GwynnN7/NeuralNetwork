#include "cli.hpp"

#include "CLI/CLI.hpp"
#include "types.hpp"

#include <print>

Args Args::parse(int argc, char* argv[]) {
    Args args;
    CLI::App app{"Neural Network Training"};
    app.add_option("dataset", args.dataset_type, "Dataset type")
        ->transform(CLI::CheckedTransformer(std::map<std::string, DatasetType>{{"xor", DatasetType::XOR}, {"xor_hot", DatasetType::XOR_HOT}, {"mnist", DatasetType::MNIST}}, CLI::ignore_case))
        ->required();

    app.add_option("--name", args.name, "Name for the model and log files")->default_val("model");
    app.add_option("--params", args.model_file, "File containing model parameters for grid search")->default_val("artifacts/grid.csv");
    app.add_option("--inner-k", args.inner_folds, "Number of folds for inner cross-validation")->default_val(1);
    app.add_option("--outer-k", args.outer_folds, "Number of folds for outer cross-validation")->default_val(1);

    app.add_option("--epochs", args.epochs, "Number of epochs")->default_val(500);
    app.add_flag("--shuffle", args.shuffle, "Shuffle dataset before splitting into folds");
    app.add_flag("--dump", args.dump, "Dump best models' weights to file after total retraining")->default_val(false);
    app.add_flag("--train", args.train, "Train a new model")->default_val(false);

    app.add_option("--train_ratio", args.train_ratio, "Training set ratio")->default_val(0.85);
    app.add_option("--dataset_ratio", args.dataset_ratio, "Subset of dataset used (when applicable)")->default_val(1.0);

    app.add_option("--seed", args.seed, "Random seed")->default_val(42);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        std::println(stderr, "Error parsing arguments: {}", e.what());
        throw std::invalid_argument("Invalid command line arguments.");
    }

    return args;
}