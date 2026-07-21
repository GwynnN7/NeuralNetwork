#pragma once

#include "cli.hpp"
#include "network.hpp"

void dump(const std::string& filename, const Args& args, Network* network);
Network* load_model(const std::string& filename, Args* args);