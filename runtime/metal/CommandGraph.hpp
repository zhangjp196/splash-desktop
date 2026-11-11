#pragma once

#include "MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::metal {

// An ordered dispatch list for one command, with the event steps that order
// it against another agent or the CPU (EventStep); MetalBackend splits a
// command into Metal command buffers at its event signals and host steps.
// Buffers bind at indices 0..n-1; an optional parameter struct binds at
// index n and is copied into the dispatch's own storage until submission.
//
// A decode cycle rebuilds its whole graph every cycle: the same kernels over
// the same tensors, with only the lanes' state bindings and parameter bytes
// different from the cycle before. Rebuilding into fresh containers would
// allocate a binding vector, a parameter buffer and a pipeline-name string
// per dispatch, hundreds of times per cycle, on the thread the GPU waits
// behind. reuseStorage() rebuilds over the previous build's storage instead:
// every dispatch slot keeps its binding vector, parameter bytes and name,
// and every add() writes the fresh values over them. A binding the rebuild
// binds again unchanged (same view of the same allocation) keeps the buffer
// it already holds, which saves the shared-pointer traffic of the weights
// every layer binds; everything else is overwritten, so a rebuilt graph is
// byte for byte the one a fresh build would have produced, whatever the
// shape, the lanes or the plan changed in between. The rebuild may also
// throw part way (a caller's validation): the next reuseStorage() starts
// from slot zero again, and command() is only reached by a build that
// finished, which trims whatever the storage still holds past the build.
class CommandGraph final {
public:
  static constexpr uint32_t kDefaultThreads = 256;

  CommandGraph() = default;
  // Dispatches own their parameter bytes; a copy would keep pointing at the
  // source's.
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;
  CommandGraph(CommandGraph &&) noexcept = default;
  CommandGraph &operator=(CommandGraph &&) noexcept = default;

  void add(std::string_view pipeline, std::vector<MetalBuffer> buffers,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(pipeline, buffers, groups, threads);
  }

  template <class Params>
  void add(std::string_view pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    addParams(pipeline, buffers, params, groups, threads);
  }

  // A dispatch of the grid in `grid`, which an earlier dispatch writes
  // within `groups` (ComputeDispatch::indirectThreadgroups).
  template <class Params>
  void add(std::string_view pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups, IndirectGrid grid,
           DispatchSize threads) {
    addParams(pipeline, buffers, params, groups, threads)
        .indirectThreadgroups = std::move(grid);
  }

  // Event steps after the dispatches added so far (EventStep): signal once
  // all earlier work has completed, or hold all later work until the event
  // reaches value.
  void signal(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Signal);
  }
  void wait(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Wait);
  }
  // A host step after the dispatches added so far (EventStep::Kind::Host):
  // `work` runs on the CPU between the work before and the work after it,
  // `timed` gets its round trip once the command completed, `gpu` says
  // whether the GPU spins through it, and `then` runs once the work after it
  // may go.
  void host(std::function<void()> work, std::function<void(const HostStepTiming &)> timed = {},
            HostStepGpu gpu = HostStepGpu::Idle, std::function<void()> then = {}) {
    EventStep &slot = reserveEvent();
    slot.before = dispatchCursor_;
    slot.event = {};
    slot.value = 0;
    slot.kind = EventStep::Kind::Host;
    slot.work = std::move(work);
    slot.timed = std::move(timed);
    slot.gpu = gpu;
    slot.then = std::move(then);
  }

  [[nodiscard]] bool empty() const noexcept {
    return reuse_ ? dispatchCursor_ == 0 : dispatches_.empty();
  }
  // The dispatches alone; command() has the event steps too. Like command(),
  // a reuse build's answer is what it wrote, not what its storage still
  // holds past it.
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    if (reuse_) return {dispatches_.data(), dispatchCursor_};
    return dispatches_;
  }
  // Hands out the dispatches and event steps to submit. A reuse build's
  // storage may hold dispatches past it, which a shape the build before ran
  // with more of left there; the command is what this build wrote, and the
  // next reuseStorage() rebuilds over the whole storage again. A build that
  // never reaches this (it threw) leaves the graph for the next
  // reuseStorage() to rebuild over.
  [[nodiscard]] Command command() const noexcept {
    if (reuse_)
      return {{dispatches_.data(), dispatchCursor_},
              {events_.data(), eventCursor_}};
    return {dispatches_, events_};
  }

  // Begins a rebuild that writes over the previous build's storage
  // (CommandGraph): the dispatches and event steps of the build before are
  // reused slot by slot, and command() trims what this build leaves past
  // itself. Called once per rebuild, before its first add().
  void reuseStorage() noexcept {
    reuse_ = true;
    dispatchCursor_ = 0;
    eventCursor_ = 0;
  }

private:
  template <class Params>
  ComputeDispatch &addParams(std::string_view pipeline,
                             std::vector<MetalBuffer> buffers,
                             const Params &params, DispatchSize groups,
                             DispatchSize threads) {
    ComputeDispatch &dispatch = push(pipeline, buffers, groups, threads);
    // The slot's parameter storage is reused across rebuilds; a shape that
    // needs other bytes grows or shrinks it, and the binding points at
    // wherever it ended up.
    dispatch.params.resize(sizeof(Params));
    std::memcpy(dispatch.params.data(), &params, sizeof(Params));
    dispatch.bytes.push_back({static_cast<uint32_t>(dispatch.buffers.size()),
                              dispatch.params.data(), sizeof(Params)});
    return dispatch;
  }

  // The next dispatch of the build: the reused slot of the build before, or
  // a fresh one once the build runs past it. Either way every field the
  // dispatch carries is written, so nothing of the build before survives
  // into this one.
  ComputeDispatch &push(std::string_view pipeline,
                        std::vector<MetalBuffer> buffers,
                        DispatchSize groups, DispatchSize threads) {
    ComputeDispatch *dispatch;
    if (reuse_) {
      if (dispatchCursor_ == dispatches_.size()) dispatches_.emplace_back();
      dispatch = &dispatches_[dispatchCursor_++];
      // A binding the rebuild binds again unchanged keeps the buffer the
      // slot already holds: sameView names the same view of the same
      // allocation, which encodes identically, and the weights every layer
      // binds again are all of them. One that changed is replaced.
      if (dispatch->buffers.size() == buffers.size()) {
        for (size_t index = 0; index < buffers.size(); ++index) {
          MetalBuffer &held = dispatch->buffers[index].buffer;
          if (!held.sameView(buffers[index])) held = buffers[index];
        }
      } else {
        dispatch->buffers.clear();
        dispatch->buffers.reserve(buffers.size());
        for (uint32_t index = 0; index < buffers.size(); ++index)
          dispatch->buffers.push_back({index, buffers[index]});
      }
      dispatch->bytes.clear();
      dispatch->params.clear();
      dispatch->indirectThreadgroups = {};
    } else {
      ComputeDispatch fresh;
      fresh.buffers.reserve(buffers.size());
      for (uint32_t index = 0; index < buffers.size(); ++index)
        fresh.buffers.push_back({index, buffers[index]});
      dispatches_.push_back(std::move(fresh));
      dispatch = &dispatches_.back();
      ++dispatchCursor_;
    }
    dispatch->pipelineName.assign(pipeline.data(), pipeline.size());
    dispatch->threadgroups = groups;
    dispatch->threadsPerThreadgroup = threads;
    return *dispatch;
  }

  // The next event step of the build, reused or fresh as push() does. A
  // reused slot is cleared of the step it held: a host step's work and its
  // round trip belong to the build that added them.
  EventStep &reserveEvent() {
    if (reuse_) {
      if (eventCursor_ == events_.size()) events_.emplace_back();
      return events_[eventCursor_++];
    }
    events_.emplace_back();
    ++eventCursor_;
    return events_.back();
  }

  void step(SharedEvent event, uint64_t value, EventStep::Kind kind) {
    EventStep &slot = reserveEvent();
    slot.before = dispatchCursor_;
    slot.event = std::move(event);
    slot.value = value;
    slot.kind = kind;
    slot.work = {};
    slot.timed = {};
    slot.gpu = HostStepGpu::Idle;
    slot.then = {};
  }

  // The dispatches and event steps of the build in progress, and how far it
  // has written into the storage of the build before.
  bool reuse_ = false;
  size_t dispatchCursor_ = 0;
  size_t eventCursor_ = 0;
  std::vector<ComputeDispatch> dispatches_;
  std::vector<EventStep> events_;
};

} // namespace splash::metal
