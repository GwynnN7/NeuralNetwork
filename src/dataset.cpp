#include "dataset.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

Matrix load_csv(const std::string &filename) {
    std::ifstream file(filename);
    std::string line, cell;
    std::vector<Scalar> values;
    int rows = 0;
    int cols = 0;
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        int current_cols = 0;
        while (std::getline(lineStream, cell, ',')) {
            try {
                if (typeid(Scalar) == typeid(double)) {
                    values.push_back(std::stod(cell));
                } else {
                    values.push_back(std::stof(cell));
                }
            } catch (const std::invalid_argument &e) {
                throw std::runtime_error("Invalid number in dataset: " + cell);
            }
            current_cols++;
        }
        if (cols == 0) {
            cols = current_cols;
        } else if (cols != current_cols) {
            throw std::runtime_error("Inconsistent number of columns in dataset.");
        }
        rows++;
    }
    file.close();
    return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols);
}

Dataset load_dataset(DatasetType dataset_type) {
    Matrix features;
    Matrix labels;
    int num_samples, num_features, num_classes;

    switch (dataset_type) {
    case DatasetType::XOR: {
        Matrix xor_data = load_csv("dataset/xor.csv").transpose();

        features = xor_data.topRows(xor_data.rows() - 1);
        labels = xor_data.bottomRows(1);
    } break;
    case DatasetType::XOR_HOT: {
        Matrix xor_hot_data = load_csv("dataset/xor_hot.csv").transpose();

        features = xor_hot_data.topRows(xor_hot_data.rows() - 2);
        labels = xor_hot_data.bottomRows(2);
    } break;
    case DatasetType::MNIST:
        throw std::runtime_error("MNIST dataset loading not implemented yet.");
    default:
        throw std::invalid_argument("Unsupported dataset type.");
    }

    num_samples = features.cols();
    num_features = features.rows();
    num_classes = labels.rows();

    std::cout << "Dataset Info:" << "\n"
              << std::left << std::setw(25) << " • Samples:" << num_samples << "\n"
              << std::left << std::setw(25) << " • Features:" << num_features << "\n"
              << std::left << std::setw(25) << " • Classes:" << num_classes << "\n\n";

    return Dataset{features, labels, num_samples, num_features, num_classes};
}