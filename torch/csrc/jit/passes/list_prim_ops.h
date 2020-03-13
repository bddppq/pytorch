#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>

namespace torch {
namespace jit {

TORCH_API std::unordered_map<NodeKind, int64_t> ListPrimOps(const std::shared_ptr<Graph>& graph);

}
} // namespace torch
