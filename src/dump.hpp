#pragma once

#include "network.hpp"

void dump_model(const std::string& file, const Model& model, Network* network);
Network* load_model(const std::string& file);