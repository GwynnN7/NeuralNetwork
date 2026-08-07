#include "cli.hpp"
#include "dataset.hpp"
#include "model.hpp"
#include "network.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <algorithm>
#include <csignal>
#include <filesystem>
#include <format>
#include <print>
#include <vector>

namespace {
// The split used for the final dump and for inference
DataSplit main_split(const Dataset& dataset, const Args& args) {
    return DataSplit::split(1, args.train_ratio, dataset.train_samples, dataset.num_samples, args.shuffle, static_cast<unsigned int>(args.seed)).front();
}
} // namespace

void train(const Dataset& dataset, const Args& args) {
    // Prepare outer folds for cross-validation and load the grid search parameters
    const std::vector<DataSplit> outer_folds = DataSplit::split(args.outer_folds, args.train_ratio, dataset.train_samples, dataset.num_samples, args.shuffle);
    std::vector<Model> grid_search = Model::load_grid_search(args.model_file);
    for (auto& grid_model : grid_search) {
        grid_model.task = dataset.task; // Assign the task type to the model based on the dataset
    }

    // Initialize a vector to store the best models for each outer fold
    std::vector<Model> best_models;
    best_models.reserve(outer_folds.size());

    // Outer loop for cross-validation
    for (size_t i = 0; i < outer_folds.size(); ++i) {
        const int outer_index = static_cast<int>(i);

        // Prepare inner folds for model selection, on training set of the current outer fold (train_valid). Pass train_samples as 0 for inner folds since it is only relevant for outer folds
        std::vector<DataSplit> inner_folds = DataSplit::split(args.inner_folds, args.train_ratio, 0, static_cast<int>(outer_folds[i].train_indices.size()), args.shuffle);
        // Remap the inner fold indices to the original dataset indices
        for (auto& inner_fold : inner_folds) {
            for (int& relative_idx : inner_fold.train_indices) {
                relative_idx = outer_folds[i].train_indices[relative_idx];
            }
            for (int& relative_idx : inner_fold.test_indices) {
                relative_idx = outer_folds[i].train_indices[relative_idx];
            }
        }
        bool model_selection = !inner_folds.empty() && grid_search.size() > 1; // Determine if model selection is needed

        // Variables to track the best model and its performance across inner folds
        size_t best_model_index = 0;
        SplitResults best_model_performance;

        // Inner loop for model selection, skipped when there is nothing to select
        for (size_t m = 0; model_selection && m < grid_search.size(); ++m) {
            const Model& grid_model = grid_search[m];
            grid_model.print(); // Print the model configuration for logging

            SplitResults current_model_performance; // Metrics for the current model across inner folds
            for (size_t j = 0; j < inner_folds.size(); ++j) {
                // Create a new network for the current model configuration
                Network network(grid_model, dataset.num_features, dataset.num_classes);

                const TrainContext ctx{
                    .epochs = args.epochs,
                    .patience = args.patience,
                    .warmup = args.warmup,
                    .model_id = grid_model.id,
                    .outer_index = outer_index,
                    .inner_index = static_cast<int>(j),
                    .in_model_selection = true,
                    .logging = true,
                };

                // Train the model on the current inner fold
                const SplitResults train_results = network.train(dataset, inner_folds[j], ctx);
                current_model_performance.add(train_results); // Accumulate metrics for the current model across inner folds
            }
            current_model_performance.average(); // Average the metrics across inner folds for the current model to get a single performance metric

            // Compare the current model's performance with the best model's performance and update if it's better
            if (m == 0 || current_model_performance.is_better_than(best_model_performance)) {
                best_model_index = m;
                best_model_performance = current_model_performance;
            }
        }

        best_models.push_back(grid_search[best_model_index]);

        if (model_selection) {
            std::println("\nRetraining the best model for Outer Fold {}: Model {}", outer_index, best_models[i].id);
        }
        best_models[i].print();

        // (Re)Train the best model of each outer fold on the train(_valid) set
        Network best_fold_network(best_models[i], dataset.num_features, dataset.num_classes);
        const TrainContext ctx{
            .epochs = args.epochs,
            .patience = args.patience,
            .warmup = args.warmup,
            .model_id = best_models[i].id,
            .outer_index = outer_index,
            .inner_index = -1,
            .in_model_selection = false,
            .logging = true,
        };
        best_fold_network.train(dataset, outer_folds[i], ctx);
    }

    std::println("\n[Best Models Summary]");
    for (size_t i = 0; i < best_models.size(); ++i) {
        std::println(" • Fold {}: Model {}", i, best_models[i].id);
    }

    if (args.dump) {
        std::print("\n[Dumping Best Models]");
        // Dump the best models found with cross-validation, trained on the original train/test split (if applicable)
        const DataSplit split = main_split(dataset, args);
        for (size_t i = 0; i < best_models.size(); ++i) {
            Network final_network(best_models[i], dataset.num_features, dataset.num_classes);
            const TrainContext ctx{
                .epochs = args.epochs,
                .patience = args.patience,
                .warmup = args.warmup,
                .model_id = best_models[i].id,
                .outer_index = static_cast<int>(i),
                .inner_index = -1,
                .in_model_selection = false,
                .logging = false,
            };
            final_network.train(dataset, split, ctx);
            Serializer::dump_model(std::format("{}/outer{}.bin", MODEL_PATH, i), best_models[i], final_network);
        }
    }
}

void test(const Dataset& dataset, const Args& args) {
    if (!std::filesystem::is_directory(MODEL_PATH)) {
        throw std::runtime_error("No artifact directory to load models from: " + MODEL_PATH);
    }

    // Collect the .bin model files and sort them
    std::vector<std::filesystem::path> model_files;
    for (const auto& file : std::filesystem::directory_iterator(MODEL_PATH)) {
        if (file.path().extension() == ".bin") {
            model_files.push_back(file.path());
        }
    }
    std::ranges::sort(model_files);

    if (model_files.empty()) {
        throw std::runtime_error("No .bin model files found in " + MODEL_PATH + " (run with --train --dump first)");
    }

    // Get the main split for evaluation
    const DataSplit split = main_split(dataset, args);

    // Extract training and testing features and labels for the split
    const Matrix train_features = dataset.features(Eigen::placeholders::all, split.train_indices);
    const Matrix train_labels = dataset.labels(Eigen::placeholders::all, split.train_indices);
    const Matrix test_features = dataset.features(Eigen::placeholders::all, split.test_indices);
    const Matrix test_labels = dataset.labels(Eigen::placeholders::all, split.test_indices);

    // Load the best models trained and evaluate them
    for (const auto& file : model_files) {
        std::unique_ptr<Network> network = Serializer::load_model(file, dataset);
        if (network == nullptr) {
            std::println(stderr, "Failed to load the model, skipping");
            continue;
        }

        // Evaluate the model
        const Matrix final_train_predictions = network->predict(train_features);
        const Matrix final_test_predictions = network->predict(test_features);

        // Calculate metrics for the split and print them
        const bool track_accuracy = (network->model.task == TaskType::CLASSIFICATION);
        SplitResults results;
        results.train_metrics.append(train_labels, final_train_predictions, network->getLossFunction(), track_accuracy);
        results.test_metrics.append(test_labels, final_test_predictions, network->getLossFunction(), track_accuracy);
        results.print(network->model.task);
    }
}

int main(int argc, char* argv[]) {
    std::println("\n\n[Neural Network Framework]");

    std::signal(SIGUSR1, handle_signal);

    try {
        Args args = Args::parse(argc, argv);

        MODEL_PATH = "artifacts/" + args.name;
        std::filesystem::create_directories(MODEL_PATH);

        // Set the random seed for reproducibility
        set_random_seed(static_cast<unsigned int>(args.seed));

        // Load the dataset
        Dataset dataset = Dataset::load(args.dataset_type, args.dataset_ratio);
        dataset.print_info();

        if (args.train) {
            train(dataset, args);
        } else {
            test(dataset, args);
        }
    } catch (const std::exception& e) {
        std::println(stderr, "\n[Fatal Error] {}", e.what());
        return 1;
    }

    return 0;
}
