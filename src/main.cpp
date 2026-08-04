#include "cli.hpp"
#include "dataset.hpp"
#include "model.hpp"
#include "network.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <csignal>
#include <filesystem>
#include <print>
#include <vector>

void train(const Dataset& dataset, const Args& args) {
    // Prepare outer folds for cross-validation and load the grid search parameters
    const std::vector<DataSplit> outer_folds = DataSplit::split(dataset.num_samples, args.outer_folds, args.train_ratio, dataset.original_num_train_samples, args.shuffle);
    std::vector<Model> grid_search = Model::load_grid_search(args.model_file);
    std::map<int, Model> best_models;

    // Outer loop for cross-validation
    for (size_t i = 0; i < outer_folds.size(); ++i) {
        // Keep track of the best model for the current outer fold
        SplitResults best_model_performance;

        // Prepare inner folds for model selection, on training set of the current outer fold (train_valid). Pass original_train_samples as 0 for inner folds since it is only relevant for outer folds
        std::vector<DataSplit> inner_folds = DataSplit::split(outer_folds[i].train_indices.size(), args.inner_folds, args.train_ratio, 0, args.shuffle);
        // Remap the inner fold indices to the original dataset indices
        for (auto& inner_fold : inner_folds) {
            for (int& relative_idx : inner_fold.train_indices) {
                relative_idx = outer_folds[i].train_indices[relative_idx];
            }
            for (int& relative_idx : inner_fold.test_indices) {
                relative_idx = outer_folds[i].train_indices[relative_idx];
            }
        }
        // Inner loop for model selection
        for (auto& grid_model : grid_search) {
            grid_model.task = dataset.task;         // Assign the task type to the model based on the dataset
            SplitResults current_model_performance; // Metrics for the current model across inner folds
            // Loop through inner folds if in model selection
            for (size_t j = 0; j < inner_folds.size() && grid_search.size() > 1; ++j) {
                // Create a new network for the current model configuration
                Network network(grid_model, dataset.num_features, dataset.num_classes);

                // Train the model on the current inner fold
                const SplitResults train_results = network.train(dataset, inner_folds[j], args.epochs, args.patience, grid_model.id, i, j);
                current_model_performance.add(train_results); // Accumulate metrics for the current model across inner folds
            }
            current_model_performance.average(); // Average the metrics across inner folds for the current model to get a single performance metric

            // Compare the current model's performance with the best model's performance and update if it's better
            if (current_model_performance > best_model_performance) {
                best_models.insert_or_assign(i, grid_model);
                best_model_performance = current_model_performance;
            }
        }

        // (Re)Train the best model of each outer fold on the train(_valid) set
        Network best_fold_network = Network(best_models[i], dataset.num_features, dataset.num_classes);
        best_fold_network.train(dataset, outer_folds[i], args.epochs, args.patience, best_models[i].id, i, -1); // Pass -1 for inner_index to indicate no model selection during final training
    }

    std::println("\n[Best Models Summary]");
    for (const auto& [outer_index, best_model] : best_models) {
        std::println(" • Fold {}: Model {}", outer_index, best_model.id);
    }
    if (args.dump) {
        std::print("\n[Dumping Best Models]");
        // Dump the best models found with cross-validation, trained on the original train/test split (if applicable)
        const DataSplit main_split = DataSplit::split(dataset.num_samples, 1, args.train_ratio, dataset.original_num_train_samples, args.shuffle).front();
        for (size_t i = 0; i < outer_folds.size(); ++i) {
            Network final_network(best_models[i], dataset.num_features, dataset.num_classes);
            final_network.train(dataset, main_split, args.epochs, args.patience, best_models[i].id, i, -1, false);
            Serializer::dump_model(std::format("{}/outer{}.bin", MODEL_PATH, i), best_models[i], &final_network);
        }
    }
}

void test(const Dataset& dataset, const Args& args) {
    // Load the best models trained and evaluate them
    for (const auto& file : std::filesystem::directory_iterator(MODEL_PATH)) {
        if (file.path().extension() != ".bin") {
            continue;
        }
        std::unique_ptr<Network> network(Serializer::load_model(file.path(), dataset));
        if (network == nullptr) {
            std::println(stderr, "Failed to load the model, skipping");
            continue;
        }
        // Evaluate the model on the original train/test split (if applicable)
        const DataSplit main_split = DataSplit::split(dataset.num_samples, 1, args.train_ratio, dataset.original_num_train_samples, args.shuffle).front();

        SplitResults results;
        MetricsResult train_metrics, test_metrics;

        // Extract training and testing features and labels for the current split
        Matrix train_features = dataset.features(Eigen::placeholders::all, main_split.train_indices);
        Matrix train_labels = dataset.labels(Eigen::placeholders::all, main_split.train_indices);
        Matrix test_features = dataset.features(Eigen::placeholders::all, main_split.test_indices);
        Matrix test_labels = dataset.labels(Eigen::placeholders::all, main_split.test_indices);

        // Evaluate the model
        Matrix final_train_predictions = network->predict(train_features);
        Matrix final_test_predictions = network->predict(test_features);

        // Calculate metrics for the split and print them
        train_metrics.append(train_labels, final_train_predictions, network->getLossFunction());
        test_metrics.append(test_labels, final_test_predictions, network->getLossFunction());
        results.add({train_metrics, test_metrics});
        results.print(network->model.task);
    }
}

int main(int argc, char* argv[]) {
    std::println("\n\n[Neural Network Framework]");

    std::signal(SIGUSR1, handle_signal);
    Args args = Args::parse(argc, argv);

    MODEL_PATH = "artifacts/" + args.name;
    std::filesystem::create_directories(MODEL_PATH);

    // Set the random seed for reproducibility
    set_random_seed(args.seed);

    // Load the dataset
    Dataset dataset = Dataset::load(args.dataset_type, args.dataset_ratio);

    if (args.train) {
        train(dataset, args);
    } else {
        test(dataset, args);
    }
    return 0;
}