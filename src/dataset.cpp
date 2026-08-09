#include "dataset.hpp"

#include "types.hpp"
#include "utility.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <istream>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Loader {

// One-hot encode the raw labels into a binary matrix
Matrix one_hot_encode(const Matrix& raw_labels, int num_classes) {
    // Initialize a zero matrix for one-hot encoding
    Matrix one_hot_labels = Matrix::Zero(raw_labels.size(), num_classes);
    for (Eigen::Index i = 0; i < raw_labels.size(); ++i) {
        const int label = static_cast<int>(raw_labels(i));
        if (label < 0 || label >= num_classes) {
            throw std::out_of_range("Label " + std::to_string(label) + " is out of range for one-hot encoding with " + std::to_string(num_classes) + " classes");
        }
        // Set the corresponding index to 1 for one-hot encoding
        one_hot_labels(i, label) = 1.0;
    }
    return one_hot_labels;
}

// Build a ColMajor Matrix from row-major data, so the rest of the framework can use it
Matrix to_matrix(const std::vector<Scalar>& flat, int rows, int cols) {
    auto row_major = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(flat.data(), rows, cols);
    return Matrix(row_major);
}

// MNIST utility functions for reading data
namespace MNIST {
// MNIST file magic numbers
constexpr int IMAGE_MAGIC = 2051;
constexpr int LABEL_MAGIC = 2049;

// MNIST files are big-endian, need to swap on little-endian systems
std::int32_t swap_endian(std::int32_t value) {
    if constexpr (std::endian::native == std::endian::little) {
        return static_cast<std::int32_t>(std::byteswap(static_cast<std::uint32_t>(value)));
    }
    return value;
}

// Read 'size' bytes from the input stream
void read_bytes(std::istream& in, void* dest, std::size_t size, const std::string& path, const char* tag) {
    in.read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(size));
    // Check if the expected number of bytes was read
    if (!in || in.gcount() != static_cast<std::streamsize>(size)) {
        throw std::runtime_error("Error for file " + path + " while reading " + tag);
    }
}

// Read a big-endian integer from the input stream
int read_int(std::istream& in, const std::string& path, const char* tag) {
    int value = 0;
    read_bytes(in, &value, sizeof(value), path, tag);
    return swap_endian(value);
}

// Take a subset of the total number of samples based on the dataset_ratio
int sample_subset(int total, Scalar dataset_ratio) {
    if (total <= 0) {
        throw std::runtime_error("Invalid total number of samples: " + std::to_string(total));
    }
    const int scaled = static_cast<int>(total * dataset_ratio);
    return std::max(1, std::min(scaled, total));
}

// Load MNIST images from the specified file path and normalize pixel values to [0, 1]
Matrix load_mnist_images(const std::string& path, Scalar dataset_ratio) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read the magic number and validate it
    if (MNIST::read_int(file, path, "magic number") != MNIST::IMAGE_MAGIC) {
        throw std::runtime_error("Invalid MNIST image file: " + path);
    }

    // Read the number of images, rows, and columns
    const int total_images = MNIST::read_int(file, path, "image count");
    const int num_rows = MNIST::read_int(file, path, "row count");
    const int num_cols = MNIST::read_int(file, path, "column count");
    if (num_rows <= 0 || num_cols <= 0) {
        throw std::runtime_error("Invalid MNIST image dimensions in " + path);
    }

    // Define a subset of images based on the dataset ratio
    const int num_images = MNIST::sample_subset(total_images, dataset_ratio);
    const int image_size = num_rows * num_cols;

    // Read the raw pixel data
    std::vector<unsigned char> raw_pixels(static_cast<std::size_t>(num_images) * image_size);
    MNIST::read_bytes(file, raw_pixels.data(), raw_pixels.size(), path, "pixel data");

    // Normalize pixel values to the range [0, 1] and store them in a matrix
    Matrix normalized_images(num_images, image_size);
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            normalized_images(i, j) = static_cast<Scalar>(raw_pixels[static_cast<std::size_t>(i) * image_size + j]) / Scalar(255);
        }
    }
    return normalized_images;
}

// Load MNIST labels from the specified file path and convert them to one-hot encoding
Matrix load_mnist_labels(const std::string& path, Scalar dataset_ratio) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read the magic number and validate it
    if (MNIST::read_int(file, path, "magic number") != MNIST::LABEL_MAGIC) {
        throw std::runtime_error("Invalid MNIST label file: " + path);
    }

    // Read the number of labels and define a subset based on the dataset ratio
    const int total_labels = MNIST::read_int(file, path, "label count");
    const int num_labels = MNIST::sample_subset(total_labels, dataset_ratio);

    // Read the raw label data
    std::vector<unsigned char> raw_labels(static_cast<std::size_t>(num_labels));
    MNIST::read_bytes(file, raw_labels.data(), raw_labels.size(), path, "label data");

    // Convert the unsigned char vector into an Eigen Scalar Matrix
    Matrix scalar_labels(num_labels, 1);
    for (int i = 0; i < num_labels; ++i) {
        scalar_labels(i, 0) = static_cast<Scalar>(raw_labels[i]);
    }

    // Convert to one-hot encoding
    return one_hot_encode(scalar_labels, 10);
}
} // namespace MNIST

namespace XOR {
constexpr int NUM_FEATURES = 2; // Number of features in the XOR dataset
constexpr int NUM_LABELS = 1;   // Number of labels in the XOR dataset
constexpr int NUM_CLASSES = 2;  // Number of classes in the XOR dataset
constexpr int XOR_NUM_COLS = NUM_FEATURES + NUM_LABELS;

// Load XOR dataset from a CSV file: 2 feature columns followed by the label column
Matrix load_xor(const std::string& filename) {
    const std::vector<CsvRow> rows = load_csv(filename);

    std::vector<Scalar> flat;
    flat.reserve(rows.size() * XOR_NUM_COLS);
    for (const CsvRow& row : rows) {
        if (static_cast<int>(row.cells.size()) != XOR_NUM_COLS) {
            throw std::runtime_error("Wrong number of columns in XOR dataset file " + filename);
        }
        for (const std::string& cell : row.cells) {
            flat.push_back(parse_cell<Scalar>(cell, filename, row.line));
        }
    }
    return to_matrix(flat, static_cast<int>(rows.size()), XOR_NUM_COLS);
}
} // namespace XOR

namespace MONK {
constexpr int NUM_FEATURES = 6;                                // Number of categorical features in the MONK dataset
constexpr int NUM_LABELS = 1;                                  // Number of labels in the MONK dataset
constexpr int NUM_CLASSES = 2;                                 // Number of classes in the MONK dataset
constexpr int NUM_SKIP_COLS = 1;                               // Number of columns to skip in the MONK file (the ID column)
const std::vector<int> feature_hot_sizes = {3, 3, 2, 3, 4, 2}; // Sizes of the one-hot encoded features for each categorical feature

// Load MONK dataset from file
Matrix load_monk(const std::string& filename, Scalar dataset_ratio) {
    const int one_hot_width = std::ranges::fold_left(feature_hot_sizes, 0, std::plus{});
    const int out_cols = NUM_LABELS + one_hot_width;

    const std::vector<CsvRow> rows = load_csv(filename, ' ');

    std::vector<Scalar> flat;
    flat.reserve(rows.size() * out_cols);
    for (const CsvRow& row : rows) {
        if (static_cast<int>(row.cells.size()) != NUM_LABELS + NUM_FEATURES + NUM_SKIP_COLS) {
            throw std::runtime_error("Wrong number of columns in MONK dataset file " + filename);
        }

        // Column 0 is the class label
        flat.push_back(parse_cell<Scalar>(row.cells[0], filename, row.line));

        // Columns 1-6 are the categorical features
        for (int col = 0; col < NUM_FEATURES; ++col) {
            const int width = feature_hot_sizes[col];
            const int value = static_cast<int>(parse_cell<Scalar>(row.cells[col + 1], filename, row.line));
            if (value < 1 || value > width) {
                throw std::runtime_error(filename + " line " + std::to_string(row.line) + ": attribute a" + std::to_string(col + 1) +
                                         " is " + std::to_string(value) + ", expected 1.." + std::to_string(width));
            }
            for (int j = 1; j <= width; ++j) {
                flat.push_back(j == value ? Scalar(1) : Scalar(0));
            }
        }
    }

    const int total_rows = static_cast<int>(rows.size());
    const int kept_rows = std::max(1, std::min(total_rows, static_cast<int>(total_rows * dataset_ratio)));
    return to_matrix(flat, total_rows, out_cols).topRows(kept_rows);
}
} // namespace MONK

namespace MLCUP {
constexpr int NUM_FEATURES = 12; // Number of features in the MLCUP dataset
constexpr int NUM_LABELS = 4;    // Number of labels in the MLCUP dataset
constexpr int NUM_SKIP_COLS = 1; // Number of columns to skip in the MLCUP file (the ID column)

Matrix load_mlcup(const std::string& filename, int columns) {
    const std::vector<CsvRow> rows = load_csv(filename);

    std::vector<Scalar> flat;
    flat.reserve(rows.size() * columns);
    for (const CsvRow& row : rows) {
        if (static_cast<int>(row.cells.size()) != columns + NUM_SKIP_COLS) {
            throw std::runtime_error("Wrong number of columns in MLCUP dataset file " + filename);
        }
        for (const std::string& cell : row.cells) {
            // The id column is the first column
            if (&cell == &row.cells[0]) {
                continue;
            }
            flat.push_back(parse_cell<Scalar>(cell, filename, row.line));
        }
    }
    return to_matrix(flat, static_cast<int>(rows.size()), columns);
}
} // namespace MLCUP
} // namespace Loader

void Dataset::print_info() const {
    std::println("\nDataset Info:");
    std::println("{:<25}{}", " • Type:", Lookup::name_of(Lookup::datasets, type));
    std::println("{:<25}{}", " • Task:", Lookup::name_of(Lookup::tasks, task));
    std::println("{:<25}{}", " • Samples:", num_samples);
    std::println("{:<25}{}", " • Features:", num_features);
    std::println("{:<25}{}", " • Classes:", num_classes);
}

Dataset Dataset::load(DatasetType dataset_type, Scalar dataset_ratio) {
    if (!(dataset_ratio > 0) || dataset_ratio > 1) {
        throw std::invalid_argument("dataset_ratio must be in (0, 1]");
    }
    switch (dataset_type) {
    case DatasetType::XOR:
    case DatasetType::XOR_HOT: {
        Matrix xor_data = Loader::XOR::load_xor("dataset/xor/xor.csv").transpose();

        // Split the data into features and labels
        Matrix features = xor_data.topRows(xor_data.rows() - Loader::XOR::NUM_LABELS);
        Matrix labels = dataset_type == DatasetType::XOR ? xor_data.bottomRows(Loader::XOR::NUM_LABELS).eval()
                                                         : Loader::one_hot_encode(xor_data.bottomRows(Loader::XOR::NUM_LABELS), Loader::XOR::NUM_CLASSES).transpose();

        // Replicate the dataset to create a larger dataset for training and testing (task set to regression for XOR and classification for XOR_HOT)
        return Dataset{dataset_type, dataset_type == DatasetType::XOR ? TaskType::REGRESSION : TaskType::CLASSIFICATION, features.replicate(1, 100), labels.replicate(1, 100)};
    };
    case DatasetType::MONK1:
    case DatasetType::MONK1_HOT:
    case DatasetType::MONK2:
    case DatasetType::MONK2_HOT:
    case DatasetType::MONK3:
    case DatasetType::MONK3_HOT: {
        // Extract the dataset number from the enum name and search for the corresponding MONK dataset file
        const std::string_view monk_name = Lookup::name_of(Lookup::datasets, dataset_type);
        const size_t digit_pos = monk_name.find_first_of("0123456789");
        if (digit_pos == std::string::npos) {
            throw std::invalid_argument(std::format("MONK dataset name has no problem number: {}", monk_name));
        }
        const char monk_dataset_number = monk_name[digit_pos];
        Matrix monk_train = Loader::MONK::load_monk(std::format("dataset/monk/monks-{}.train", monk_dataset_number), dataset_ratio).transpose();
        Matrix monk_test = Loader::MONK::load_monk(std::format("dataset/monk/monks-{}.test", monk_dataset_number), dataset_ratio).transpose();
        Matrix monk_data(monk_train.rows(), monk_train.cols() + monk_test.cols());
        monk_data << monk_train, monk_test;

        Matrix features = monk_data.bottomRows(monk_data.rows() - Loader::MONK::NUM_LABELS);
        const bool is_hot_encoded = (dataset_type == DatasetType::MONK1_HOT || dataset_type == DatasetType::MONK2_HOT || dataset_type == DatasetType::MONK3_HOT);
        Matrix labels = !is_hot_encoded ? monk_data.topRows(Loader::MONK::NUM_LABELS).eval()
                                        : Loader::one_hot_encode(monk_data.topRows(Loader::MONK::NUM_LABELS), Loader::MONK::NUM_CLASSES).transpose();

        return Dataset{dataset_type, TaskType::CLASSIFICATION, std::move(features), std::move(labels), static_cast<int>(monk_train.cols())};
    };
    case DatasetType::MNIST: {
        Matrix train_features = Loader::MNIST::load_mnist_images("dataset/mnist/train-images-idx3-ubyte", dataset_ratio).transpose();
        Matrix train_labels = Loader::MNIST::load_mnist_labels("dataset/mnist/train-labels-idx1-ubyte", dataset_ratio).transpose();
        Matrix test_features = Loader::MNIST::load_mnist_images("dataset/mnist/t10k-images-idx3-ubyte", dataset_ratio).transpose();
        Matrix test_labels = Loader::MNIST::load_mnist_labels("dataset/mnist/t10k-labels-idx1-ubyte", dataset_ratio).transpose();

        // Combine training and testing datasets into a single dataset for cross-validation
        Matrix all_features(train_features.rows(), train_features.cols() + test_features.cols());
        Matrix all_labels(train_labels.rows(), train_labels.cols() + test_labels.cols());

        all_features << train_features, test_features;
        all_labels << train_labels, test_labels;

        // Return the combined dataset with task set to classification for MNIST
        const int mnist_train_samples = static_cast<int>(train_features.cols());
        return Dataset{DatasetType::MNIST, TaskType::CLASSIFICATION, std::move(all_features), std::move(all_labels), mnist_train_samples};
    };
    case DatasetType::MLCUP: {
        Matrix mlcup_data = Loader::MLCUP::load_mlcup("dataset/mlcup/ML-CUP25-TR.csv", Loader::MLCUP::NUM_FEATURES + Loader::MLCUP::NUM_LABELS).transpose();
        Matrix features = mlcup_data.topRows(mlcup_data.rows() - Loader::MLCUP::NUM_LABELS);
        Matrix labels = mlcup_data.bottomRows(Loader::MLCUP::NUM_LABELS);
        // Load the blind test set, which has no labels
        Matrix blind = Loader::MLCUP::load_mlcup("dataset/mlcup/ML-CUP25-TS.csv", Loader::MLCUP::NUM_FEATURES).transpose();
        return Dataset{DatasetType::MLCUP, TaskType::REGRESSION, std::move(features), std::move(labels), std::nullopt, std::move(blind)};
    };
    default: {
        throw std::invalid_argument("Unsupported dataset type");
    }
    }
}