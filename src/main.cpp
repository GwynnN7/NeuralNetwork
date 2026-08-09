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
#include <numeric>
#include <optional>
#include <print>
#include <vector>

void train(const Dataset& dataset, const Args& args) {
    // Every split for this run comes from here, so the dataset and the options behind them stay visible
    const Splitter splitter{dataset, args};

    // Load the grid search parameters and assign the task type to the model based on the dataset
    std::vector<Model> grid_search = Model::load_grid_search(args.grid_file);
    std::ranges::for_each(grid_search, [&dataset](Model& model) { model.task = dataset.task; });

    // Initialize outer folds variables
    const std::vector<DataSplit> outer_folds = splitter.get(args.outer_folds);
    std::vector<Model> best_models;
    best_models.reserve(outer_folds.size());
    // Initialize a vector to store the summary of the best model of each outer fold
    std::vector<SplitSummary> assessment_summaries(outer_folds.size());

    // Outer loop for cross-validation
    for (size_t i = 0; i < outer_folds.size(); ++i) {
        const int outer_index = static_cast<int>(i);

        // Prepare inner folds for model selection, on training set of the current outer fold (train_valid)
        std::vector<DataSplit> inner_folds = splitter.get(args.inner_folds, &outer_folds[i].train_indices);
        bool in_model_selection = grid_search.size() > 1;

        // Variable to track the best model (or the only one) and its score
        Model best_model = grid_search.front();

        // Inner loop for model selection, skipped when there is nothing to select
        for (size_t m = 0; in_model_selection && m < grid_search.size(); ++m) {
            // Get next model and print its configuration
            const Model& grid_model = grid_search[m];
            grid_model.print();

            // Run the model on each inner fold and collect scores for selection and summary for statistics
            std::vector<SelectionScore> model_scores;
            SplitSummary model_summary;
            for (size_t j = 0; j < inner_folds.size(); ++j) {
                for (int t = 0; t < args.trials; ++t) { // Loop over trials for the current inner fold
                    set_trial_seed(args.seed, t, j);    // Seed the trial generator for reproducibility

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
                    model_scores.push_back(SelectionScore::from_run(run));
                    model_summary.add_trial(run);
                }
            }

            model_summary.print(dataset.task == TaskType::CLASSIFICATION, "Validation");

            // Calculate the average score of the model across all inner folds and trials
            const SelectionScore average_score = SelectionScore::average_scores(model_scores);
            // Keep the model with the best final score
            if (average_score.valid() && (m == 0 || !best_model.score.valid() || average_score < best_model.score)) {
                best_model = grid_model;
                best_model.score = average_score;
                best_model.summary = model_summary;
            }
        }

        best_models.push_back(best_model);
        if (in_model_selection) {
            std::println("\nRetraining the best model for Outer Fold {}: Model {}", outer_index, best_models[i].id);
        }
        best_models[i].print();

        // (Re)train the best model on the entire training set of the outer fold
        for (int t = 0; t < args.trials; ++t) {
            set_trial_seed(args.seed, t, -1);

            Network best_fold_network(best_models[i], dataset.num_features, dataset.num_classes);
            const TrainContext ctx = [&]() {
                TrainContext context = TrainContext::from_args(args);
                context.trial = t;
                context.model_id = best_models[i].id;
                context.outer_index = outer_index;
                context.inner_index = -1;
                return context;
            }();

            const RunCurves run = best_fold_network.train(dataset, outer_folds[i], ctx);
            assessment_summaries[i].add_trial(run);
        }
    }

    // Print the summary of the best models found for each outer fold
    std::println("\n[Best Models Summary]");
    const bool track_accuracy = (dataset.task == TaskType::CLASSIFICATION);
    for (size_t i = 0; i < best_models.size(); ++i) {
        std::println("\n • Fold {}: Model {}", i, best_models[i].id);
        assessment_summaries[i].print(track_accuracy, "Test");
    }

    // Dump the best model
    if (args.dump) {
        std::print("\n[Dumping Final Model]");
        set_trial_seed(args.seed, 0, -1);

        const Model& final_model = best_models.front();
        if (std::ranges::any_of(best_models, [&](const Model& m) { return m.id != final_model.id; })) {
            std::println("\nOuter folds disagreed on the configuration; using the one from fold 0 (Model {})", final_model.id);
        }

        DataSplit split;
        const int training_samples = dataset.train_samples.value_or(dataset.num_samples);
        split.train_indices.resize(training_samples);
        std::iota(split.train_indices.begin(), split.train_indices.end(), 0);

        const Stats target_error = final_model.summary.training.get(&Metrics::error);
        switch (args.stopping_rule) {
        case StoppingRule::ERROR_LEVEL:
            if (target_error.mean > 0) {
                std::println("\nStopping the final retrain at the validated training error level: {:.3f}", target_error.mean);
                break;
            }
            // Fall through to convergence if there is no validated error
            [[fallthrough]];
        case StoppingRule::CONVERGENCE:
            std::println("\nStopping the final retrain at convergence");
            break;
        }

        Network final_network(final_model, dataset.num_features, dataset.num_classes);
        const TrainContext ctx = [&]() {
            TrainContext context = TrainContext::from_args(args);
            context.model_id = final_model.id;
            context.inner_index = -1;
            context.target_error = target_error.mean;
            context.logging = false;
            return context;
        }();

        const RunCurves final_run = final_network.train(dataset, split, ctx);
        Serializer::dump_model(std::format("{}/final.bin", MODEL_PATH), final_model, final_network);
    }
}

void write_blind_predictions(const Matrix& predictions, const DatasetType dataset_type) {
    const std::filesystem::path file = std::format("{}/{}.csv", MODEL_PATH, Lookup::name_of(Lookup::datasets, dataset_type));
    std::ofstream out(file);
    if (!out.is_open()) {
        throw std::runtime_error("Could not open " + file.string() + " for writing");
    }
    const auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    out << "# \n";
    out << "# " << Lookup::name_of(Lookup::datasets, dataset_type) << "\n";
    out << "# " << std::format("{:%d %b %Y}", now) << "\n";
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
    const int training_samples = dataset.train_samples.value_or(dataset.num_samples);
    const bool has_test_split = training_samples < dataset.num_samples;
    const bool predict_blind = !has_test_split && dataset.blind_features.has_value();

    std::vector<int> evaluation_indices;
    if (has_test_split) {
        evaluation_indices.resize(dataset.num_samples - training_samples);
        std::iota(evaluation_indices.begin(), evaluation_indices.end(), training_samples);
    } else if (!predict_blind) {
        evaluation_indices.resize(dataset.num_samples);
        std::iota(evaluation_indices.begin(), evaluation_indices.end(), 0);
    }

    std::vector<int> training_indices(training_samples);
    std::iota(training_indices.begin(), training_indices.end(), 0);
    const Matrix train_features = dataset.features(Eigen::placeholders::all, training_indices);
    const Matrix train_labels = dataset.labels(Eigen::placeholders::all, training_indices);
    const Matrix test_features = dataset.features(Eigen::placeholders::all, evaluation_indices);
    const Matrix test_labels = dataset.labels(Eigen::placeholders::all, evaluation_indices);

    for (const auto& file : models) {
        const std::expected<std::unique_ptr<Network>, std::string> loaded = Serializer::load_model(file, dataset);
        if (!loaded) {
            std::println(stderr, "Skipping {}: {}", file.filename().string(), loaded.error());
            continue;
        }
        const std::unique_ptr<Network>& network = *loaded;

        if (predict_blind) {
            write_blind_predictions(network->predict(*dataset.blind_features), dataset.type);
            continue;
        }

        const bool track_accuracy = (network->model.task == TaskType::CLASSIFICATION);
        const LossType loss = network->model.loss_type;

        SplitSummary summary;
        summary.add_trial(Metrics::evaluate(train_labels, network->predict(train_features), loss, track_accuracy),
                          Metrics::evaluate(test_labels, network->predict(test_features), loss, track_accuracy));
        summary.print(track_accuracy, has_test_split ? "Test" : "Dataset");
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
        std::println(stderr, "\n[Fatal Error] {}", e.what());
        return 1;
    }

    return 0;
}
