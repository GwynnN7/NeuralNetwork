#pragma once

#include "cli.hpp"
#include "network.hpp"

void dump(const std::string& model_path, const Args& args, Network* network);
Network* load_model(const std::string& model_path, Args* args);