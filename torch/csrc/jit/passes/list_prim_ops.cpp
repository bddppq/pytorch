#include <torch/csrc/jit/passes/list_prim_ops.h>

namespace torch {
namespace jit {

namespace {

// Traverse all nodes in the block (and subblocks) and collect all
// prim:: ops
void ListPrimOps(const Block* block, std::unordered_map<NodeKind, int64_t>& stat) {
  for (const auto* node : block->nodes()) {
    if (node->kind().is_prim()) {
      auto iter = stat.find(node->kind());
      if (iter != stat.end()) {
        ++iter->second;
      } else {
        stat.insert(std::make_pair(node->kind(), 1));
      }
    }
    // Traverse sub-blocks.
    for (const auto* subblock : node->blocks()) {
      ListPrimOps(subblock, stat);
    }
  }
}
} // anonymous namespace

std::unordered_map<NodeKind, int64_t> ListPrimOps(const std::shared_ptr<Graph>& graph) {
  std::unordered_map<NodeKind, int64_t> stat{};
  ListPrimOps(graph->block(), stat);
  return stat;
}
} // namespace jit
} // namespace torch
