#include "cli.hpp"
#include "dataset.hpp"
#include "datasplit.hpp"
#include "model.hpp"
#include "network.hpp"
#include "selection.hpp"
#include "summary.hpp"
#include "types.hpp"
#include "utility.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <print>
#include <unordered_map>
#include <vector>

void train(const Dataset& dataset, const Args& args) {
    // Start time of the entire training
    const auto train_start = std::chrono::steady_clock::now();

    // Handles the splitting of the dataset into folds
    const Splitter splitter{dataset, args};

    // Load the grid search parameters and assign the task type to the model based on the dataset
    std::vector<Model> grid_search = Model::load_grid_search(args.grid_file);
    std::ranges::for_each(grid_search, [&dataset](Model& model) { model.task = dataset.task; });

    // Get the outer folds splits
    const std::vector<DataSplit> outer_folds = splitter.get(args.outer_folds);
    // Best model for each outer fold
    std::vector<Model> best_models;
    best_models.reserve(outer_folds.size());
    // Time cost of model selection
    TimingSummary selection_timing;

    // Loop for outer cross-validation (or single fold if outer_folds == 1)
    for (size_t i = 0; i < outer_folds.size(); ++i) {
        const int outer_index = static_cast<int>(i);
        if (i == 0) {
            std::println("\n---------------------------------------");
        }

        // Get the inner folds splits (not used if not in model selection, so if only one model is provided)
        std::vector<DataSplit> inner_folds = splitter.get(args.inner_folds, &outer_folds[i].train_indices);
        bool in_model_selection = grid_search.size() > 1;

        // Variable to track the best model (or the only one) and its score
        Model best_model = grid_search.front();

        // Model selection loop, skipped when there is only one model
        for (size_t m = 0; in_model_selection && !finish_flag && m < grid_search.size(); ++m) {
            // Get next model and print its configuration
            const Model& grid_model = grid_search[m];
            grid_model.print(epochs_for(args.updates, grid_model.batch_size, static_cast<int>(inner_folds.front().train_indices.size())));

            // Run the model on each inner fold and collect scores for selection, and summary for statistics
            std::vector<SelectionScore> model_scores;
            SplitSummary model_summary;
            // Loop for inner cross-validation (or single fold if outer_folds == 1)
            for (size_t j = 0; j < inner_folds.size(); ++j) {
                for (int t = 0; t < args.trials; ++t) { // Loop over trials for the current inner fold
                    set_trial_seed(args.seed, t, j);    // Seed the trial generator for reproducibility

                    // Instantiate the network for the current model and the training context
                    Network network(grid_model, dataset.num_features, dataset.num_classes);
                    const TrainContext ctx = [&]() {
                        TrainContext context = TrainContext::from_args(args);
                        context.trial = t;
                        context.model_id = grid_model.id;
                        context.outer_index = outer_index;
                        context.inner_index = static_cast<int>(j);
                        context.in_model_selection = true;
                        return context;
                    }();

                    // Train the model on the current inner fold
                    const RunCurves run = network.train(dataset, inner_folds[j], ctx);
                    model_summary.add_run(run, args.selection_window);
                }
            }

            model_summary.print(dataset.task == TaskType::CLASSIFICATION, "Split Summary", "Validation");
            // Add the duration of all inner folds and trials of this model
            selection_timing.concat(model_summary.timing);

            // Calculate the average score of the model across all inner folds and trials
            const SelectionScore average_score = model_summary.average_score();
            // Keep the model with the best final score
            if (m == 0 || (average_score < best_model.score)) {
                best_model = grid_model;
                best_model.score = average_score;
                best_model.summary = model_summary;
            }
        }

        best_models.push_back(best_model);
        if (in_model_selection) {
            std::println("\n---------------------------------------");
            std::println("Retraining Best Model {}, Outer Fold {}", best_models[i].id, outer_index);
        }
        best_models[i].print(epochs_for(args.updates, best_models[i].batch_size, static_cast<int>(outer_folds[i].train_indices.size())));

        // If the ERROR rule is used and model selection was run, the retrain will stop at the validated training error level of the inner fold that selected the best model
        // If model selection was not run, or the PATIENCE rule is used, the retrain will stop when the training error converges
        const Stats validated_error = best_models[i].summary.training.get(&Metrics::error);
        const std::optional<Scalar> outer_target_error = (args.stopping_rule == StoppingRule::ERROR && validated_error.mean > 0) ? std::optional(validated_error.mean) : std::nullopt;
        if (outer_target_error) {
            std::println("Stopping the retrain at the validated training error level: {:.3f}", *outer_target_error);
        }

        // (Re)train the best model on the entire training set of the outer fold
        best_models[i].final_summary.emplace();
        for (int t = 0; t < args.trials; ++t) {
            set_trial_seed(args.seed, t, -1);

            // Instantiate the network for the current model and the training context
            Network best_fold_network(best_models[i], dataset.num_features, dataset.num_classes);
            const TrainContext ctx = [&]() {
                TrainContext context = TrainContext::from_args(args);
                context.trial = t;
                context.model_id = best_models[i].id;
                context.outer_index = outer_index;
                context.inner_index = -1;
                context.target_error = outer_target_error;
                return context;
            }();

            const RunCurves run = best_fold_network.train(dataset, outer_folds[i], ctx);
            best_models[i].final_summary.value().add_run(run, args.selection_window);
        }

        if (finish_flag) {
            std::println("\n[Finishing early with {} outer fold(s) completed]", i + 1);
            break;
        }
    }

    if (best_models.empty()) {
        throw std::runtime_error("No models were found during training");
    }

    // --- Best Models Summary per Fold ---
    std::println("\n---------------------");
    std::println("[Best Models Summary]");
    const bool track_accuracy = (dataset.task == TaskType::CLASSIFICATION);
    TimingSummary retrain_timing;
    for (size_t i = 0; i < best_models.size(); ++i) {
        const SplitSummary& fold_summary = best_models[i].final_summary.value();
        fold_summary.print(track_accuracy, std::format("Fold {}: Model {}", i, best_models[i].id), "Test");
        retrain_timing.concat(fold_summary.timing);
    }
    // ------------------------------------

    // --- Final Assessment across Folds ---
    auto fold_ids = best_models | std::views::transform(&Model::id) | std::ranges::to<std::vector>();
    std::ranges::sort(fold_ids);
    const size_t configurations = fold_ids.size() - std::ranges::unique(fold_ids).size();

    const auto split_assessment = [&](const char* label, auto&& metrics) {
        RunSummary split_summary;
        for (const Model& model : best_models) {
            const RunSummary* summary = metrics(model);
            if (summary && !summary->empty()) {
                split_summary.add(*summary);
            }
        }
        std::println(" • {} Set:", label);
        split_summary.print(track_accuracy);
    };

    std::println("\n-------------------------------------------------------------------");
    std::println("[Final Assessment] ({} outer fold(s), {} different configuration(s))", best_models.size(), configurations);
    split_assessment("Training", [](const Model& m) { return m.final_summary ? &m.final_summary->training : nullptr; });
    split_assessment("Validation", [](const Model& m) { return &m.summary.holdout; });
    split_assessment("Test", [](const Model& m) { return m.final_summary ? &m.final_summary->holdout : nullptr; });
    // --------------------------------------

    // --- Timing Summary ---
    std::println("\n[Compute Time]");
    if (!selection_timing.empty()) {
        std::println(" • Model Selection: {:.2f}s, {} run(s)", selection_timing.total(), selection_timing.seconds.size());
    }
    if (!retrain_timing.empty()) {
        std::println(" • Best Model Retrain: {:.2f}s, {} run(s)", retrain_timing.total(), retrain_timing.seconds.size());
    }
    const Scalar entire_run = std::chrono::duration<Scalar>(std::chrono::steady_clock::now() - train_start).count();
    std::println(" • Entire Run: {} minutes and {} seconds", static_cast<int>(entire_run) / 60, static_cast<int>(entire_run) % 60);
    // ----------------------

    // Dump the best model
    if (args.dump) {
        std::println("\n[Dumping Final Model]");
        set_trial_seed(args.seed, 0, -1);

        std::unordered_map<int, int> models_occurrencies{};
        std::ranges::for_each(best_models, [&models_occurrencies](const Model& m) { models_occurrencies[m.id]++; });
        // Select the model that was most frequently selected across outer folds, or the one with the lowest id in case of a tie
        const auto most_frequent = std::ranges::max_element(models_occurrencies, [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second < b.second : a.first > b.first;
        });
        const Model& final_model = *std::ranges::find(best_models, most_frequent->first, &Model::id);

        if (std::ranges::any_of(best_models, [&](const Model& m) { return m.id != final_model.id; })) {
            std::println("Outer folds disagreed on the configuration. Using most frequent model (Model {})", final_model.id);
        }

        DataSplit split;
        // If the dataset has a predefined training split, use it. Otherwise, use the entire dataset for training
        const int training_samples = dataset.train_samples.value_or(dataset.num_samples);
        split.train_indices.resize(training_samples);
        std::iota(split.train_indices.begin(), split.train_indices.end(), 0);

        // If the ERROR rule is used, the final retrain will stop at the validated training error level of the outer fold that selected the final model
        // If the PATIENCE rule is used, the final retrain will stop when the training error converges
        const Stats target_error = final_model.final_summary ? final_model.final_summary->training.get(&Metrics::error) : Stats{};
        std::optional<Scalar> stop_at_error;
        if (args.stopping_rule == StoppingRule::ERROR && target_error.mean > 0) {
            stop_at_error = target_error.mean;
            std::println("Stopping the final retrain at the retrained training error level: {:.3f}", target_error.mean);
        } else {
            std::println("Stopping the final retrain on the {} rule", Lookup::name_of(Lookup::stopping_rules, args.stopping_rule));
        }

        // Instantiate the network for the current model and the training context
        Network final_network(final_model, dataset.num_features, dataset.num_classes);
        const TrainContext ctx = [&]() {
            TrainContext context = TrainContext::from_args(args);
            context.model_id = final_model.id;
            context.inner_index = -1;
            context.target_error = stop_at_error;
            context.logging = false;
            return context;
        }();

        // Train and dump the final model
        const RunCurves final_run = final_network.train(dataset, split, ctx);
        Serializer::dump_model(std::format("{}/final.bin", MODEL_PATH), final_model, final_network);
        std::println(" • Final Model Training: {:.2f}s, {} epochs", final_run.duration, final_run.training.size());
    }
}

constexpr const char* SUBMISSION_AUTHOR = "Gwynn7";
constexpr const char* SUBMISSION_TEAM = "gwynn7";

void write_blind_predictions(const Matrix& predictions, const DatasetType dataset_type) {
    const std::filesystem::path file = std::format("{}/{}.csv", MODEL_PATH, Lookup::name_of(Lookup::datasets, dataset_type));
    std::ofstream out(file);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open " + file.string() + " for writing");
    }

    const auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    out << "# " << SUBMISSION_AUTHOR << "\n";
    out << "# " << SUBMISSION_TEAM << "\n";
    out << "# " << (dataset_type == DatasetType::MLCUP ? "ML-CUP25" : Lookup::name_of(Lookup::datasets, dataset_type))
        << "\n";
    out << "# " << std::format("{:%d/%m/%Y}", now) << "\n";
    out << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
    for (Eigen::Index p = 0; p < predictions.cols(); ++p) {
        out << (p + 1);
        for (Eigen::Index m = 0; m < predictions.rows(); ++m) {
            out << "," << predictions(m, p);
        }
        out << "\n";
    }
    out.flush();
    if (!out) {
        throw std::runtime_error("Failed while writing " + file.string());
    }
    std::println("\n- Wrote {} predictions to {}", predictions.cols(), file.string());
}

void test(const Dataset& dataset, const Args&) {
    if (!std::filesystem::is_directory(MODEL_PATH)) {
        throw std::runtime_error("No artifact directory to load models from: " + MODEL_PATH);
    }

    // Collect the .bin model files and sort them
    std::vector<std::filesystem::path> models;
    for (const auto& file : std::filesystem::directory_iterator(MODEL_PATH)) {
        if (file.path().extension() == ".bin") {
            models.push_back(file.path());
        }
    }
    std::ranges::sort(models);

    if (models.empty()) {
        throw std::runtime_error("No .bin model files found in " + MODEL_PATH + " (run with --train --dump first)");
    }

    // Determine the features and labels to use for evaluation
    // If the dataset has a predefined training split, use it. Otherwise, use the entire dataset for training
    const int training_samples = dataset.train_samples.value_or(dataset.num_samples);
    const bool has_test_split = training_samples < dataset.num_samples;
    // If there is no default test split, check if the dataset has blind features for prediction
    const bool predict_blind = !has_test_split && dataset.blind_features.has_value();

    std::vector<int> evaluation_indices;
    if (has_test_split) {
        // If the dataset has a predefined test split, use it for evaluation
        evaluation_indices.resize(dataset.num_samples - training_samples);
        std::iota(evaluation_indices.begin(), evaluation_indices.end(), training_samples);
    } else if (!predict_blind) {
        // If there is no predefined test split and no blind features, use the entire dataset for evaluation
        // This is for example datasets like XOR, doesn't have an actual meaning since we train and test on the same data
        evaluation_indices.resize(dataset.num_samples);
        std::iota(evaluation_indices.begin(), evaluation_indices.end(), 0);
    }

    // Extract the features and labels for training and evaluation
    std::vector<int> training_indices(training_samples);
    std::iota(training_indices.begin(), training_indices.end(), 0);
    const Matrix train_features = dataset.features(Eigen::placeholders::all, training_indices);
    const Matrix train_labels = dataset.labels(Eigen::placeholders::all, training_indices);
    const Matrix test_features = dataset.features(Eigen::placeholders::all, evaluation_indices);
    const Matrix test_labels = dataset.labels(Eigen::placeholders::all, evaluation_indices);

    for (const auto& file : models) {
        // Load and instantiate the model and network
        const std::expected<std::unique_ptr<Network>, std::string> loaded = Serializer::load_model(file, dataset);
        if (!loaded) {
            std::println(stderr, "Skipping {}: {}", file.filename().string(), loaded.error());
            continue;
        }
        const std::unique_ptr<Network>& network = *loaded;

        // If the dataset has blind features, predict on them and write the predictions to a CSV file
        if (predict_blind) {
            write_blind_predictions(network->predict(*dataset.blind_features), dataset.type);
            continue;
        }

        const bool track_accuracy = (network->model.task == TaskType::CLASSIFICATION);
        const LossType loss = network->model.loss_type;

        // Evaluate the model on the training set and evaluate the performance on the test set
        SplitSummary summary;
        summary.add_run(Metrics::evaluate(train_labels, network->predict(train_features), loss, track_accuracy),
                        Metrics::evaluate(test_labels, network->predict(test_features), loss, track_accuracy));
        summary.print(track_accuracy, "Prediction Summary", has_test_split ? "Test" : "Dataset");
    }
}

int main(int argc, char* argv[]) {
    std::println("[Neural Network Framework]");

    std::signal(SIGUSR1, handle_signal);
    std::signal(SIGUSR2, handle_finish_signal);

    try {
        Args args = Args::parse(argc, argv);

        MODEL_PATH = "artifacts/" + args.name;
        std::filesystem::create_directories(MODEL_PATH);

        // Delete any existing curves logs
        if (args.train) {
            for (const auto& entry : std::filesystem::directory_iterator(MODEL_PATH)) {
                const std::string filename = entry.path().filename().string();
                if (entry.is_regular_file() && filename.starts_with("outer") && filename.contains("_m") && filename.ends_with(".csv")) {
                    std::filesystem::remove(entry.path());
                }
            }
        }

        // Set the random seed for splitting the dataset
        set_split_seed(static_cast<unsigned int>(args.seed));

        // Load the dataset
        Dataset dataset = Dataset::load(args.dataset_type, args.dataset_ratio);
        dataset.print_info();

        if (args.train) {
            train(dataset, args);
        } else {
            test(dataset, args);
        }
    } catch (const std::exception& e) {
        std::println(stderr, "\nError: {}", e.what());
        return 1;
    }

    return 0;
}
