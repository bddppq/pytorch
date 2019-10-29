#include <torch/csrc/autograd/function.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch { namespace autograd {

// autograd functions' sequence_nr is thread local.
// In order to make them less likely to collide across threads, we initialize
// them with different values:
// There is a global atomic (16bit) counter for all the threads (that ever
// create autograd functions). At the time of such a thread is created, the
// counter will be bumped and stored as the first two bytes in the initial value
// of the sequence_nr. During runtime, each sequence_nr will increase
// montonically.
// Note there are still chances the sequence_nr can collide across threads (when
// the total number of threads overflows 16bit, or in a thread the total number
// of autograd functions overflows 48bit).
namespace {
uint64_t init_next_sequence_nr() {
  static std::atomic<uint16_t> counter{0};
  return static_cast<uint64_t>(counter++)
         << ((sizeof(uint64_t) - sizeof(uint16_t)) * CHAR_BIT);
}
}

uint64_t& Node::get_next_sequence_nr() {
  thread_local static uint64_t next_sequence_nr = init_next_sequence_nr();
  return next_sequence_nr;
}

uint64_t Node::peek_at_next_sequence_nr() {
  return Node::get_next_sequence_nr();
}

auto Node::name() const -> std::string {
  return c10::demangle(typeid(*this).name());
}

AnomalyMetadata* Node::metadata() noexcept {
  if (!anomaly_metadata_) {
    anomaly_metadata_ = Engine::get_default_engine().make_anomaly_metadata();
  }
  return anomaly_metadata_.get();
}

static void gatherFunctions(
    Node* func,
    std::vector<std::shared_ptr<Node>>& stack) {
  func->release_variables();

  for (auto& edge : func->next_edges()) {
    if (edge.function.use_count() == 1) {
      stack.emplace_back(std::move(edge.function));
    } else {
      edge.function.reset();
    }
  }
}

/*
  * Fix for #5534: prevent stack overflow on deletion of deep computation graph
  *
  * Sometimes one can end up with a very big computation graph of Nodes
  * and Edges. Each std::shared_ptr<Node> contains a list of Edge, and
  * each Edge contains a std::shared_ptr<Node>. Deleting a
  * std::shared_ptr<Node> can trigger the recursive deletion of other
  * std::shared_ptr<Node>'s: this can stack overflow if the graph
  * is deep enough. Here is an example of such a graph:
  *
  * shared_ptr<Node> -> Edge -> shared_ptr<Node> -> Edge -> ... -> shared_ptr<Node>
  *
  * The solution here is to detect when we are decrementing away the last
  * reference to a Node, and when doing so to buffer up the Node's
  * that will be recursively decremented.  We can then decrement (and free)
  * the original Node without causing a recursive cascade, before
  * draining the buffer applying the same behavior.  This is, in effect,
  * converting recursion to a loop, using a heap buffer in place of the
  * recursive call stack.
  */
void deleteNode(Node* function) {
  // To avoid stack overflow on large computational graphs,
  // we need to track reference decrementing and freeing
  // on the heap.
  function->release_variables();
  std::vector<std::shared_ptr<Node>> stack;
  gatherFunctions(function, stack);
  delete function;

  while (!stack.empty()) {
    auto func = std::move(stack.back());
    stack.pop_back();
    gatherFunctions(func.get(), stack);
    // Reference count is decremented on the loop backedge.
  }
}

}} // namespace torch::autograd
