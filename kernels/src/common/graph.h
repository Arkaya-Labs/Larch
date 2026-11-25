// Stream capture into replayable graphs for the CUDA backend and its hipified HIP build (architecture
// 6.3: CUDA graphs here, HIP graphs after hipify). Only calls whose CUDA and HIP signatures match.
#pragma once

#include <cuda_runtime.h>
#include <vector>

// Captured workspace must belong to the stream, not to CUDA's persistent graph allocation pool.
// Check actual nodes after capture; a library changing its workspace behavior falls back eagerly.
// HIP's node enums vary by runtime version, so its existing graph implementation remains available.
inline cudaError_t vv_graph_bounded(cudaGraph_t root) {
#if !defined(__HIP_PLATFORM_AMD__)
  std::vector<cudaGraph_t> pending{root};
  size_t total = 0;
  while (!pending.empty()) {
    const cudaGraph_t g = pending.back();
    pending.pop_back();
    size_t n = 0;
    if (cudaError_t e = cudaGraphGetNodes(g, nullptr, &n)) return e;
    if (n > 65536 - total) return cudaErrorNotSupported;
    total += n;
    std::vector<cudaGraphNode_t> nodes(n);
    if (cudaError_t e = cudaGraphGetNodes(g, nodes.data(), &n)) return e;
    for (cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type;
      if (cudaError_t e = cudaGraphNodeGetType(node, &type)) return e;
      if (type == cudaGraphNodeTypeMemAlloc || type == cudaGraphNodeTypeMemFree)
        return cudaErrorNotSupported;
      if (type == cudaGraphNodeTypeGraph) {
        cudaGraph_t child = nullptr;
        if (cudaError_t e = cudaGraphChildGraphNodeGetGraph(node, &child)) return e;
        pending.push_back(child);
      }
    }
  }
#else
  (void)root;
#endif
  return cudaSuccess;
}

/// Starts capturing the work queued on s. Thread-local mode: other threads' CUDA calls do not break it.
inline cudaError_t vv_graph_begin(cudaStream_t s) { return cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal); }

/// Ends the capture begun on s and instantiates it into *out.
inline cudaError_t vv_graph_end(cudaStream_t s, cudaGraphExec_t* out) {
  cudaGraph_t g = nullptr;
  cudaError_t e = cudaStreamEndCapture(s, &g);
  try {
    if (e == cudaSuccess) e = vv_graph_bounded(g);
    if (e == cudaSuccess) e = cudaGraphInstantiateWithFlags(out, g, 0);
  } catch (...) {
    if (g) cudaGraphDestroy(g);
    throw;
  }
  if (g) cudaGraphDestroy(g);
  return e;
}

/// Replays g on s.
inline cudaError_t vv_graph_launch(cudaGraphExec_t g, cudaStream_t s) { return cudaGraphLaunch(g, s); }

/// Frees an instantiated graph.
inline void vv_graph_destroy(cudaGraphExec_t g) { cudaGraphExecDestroy(g); }
