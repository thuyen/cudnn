#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>


#include "THC/THC.h"

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include "ATen/native/utils/ParamsHash.h"

#include <ATen/TensorUtils.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

namespace at { namespace native {

// TODO: Go through all the checking code again and make sure
// we haven't missed anything.

// ---------------------------------------------------------------------
//
// Math
//
// ---------------------------------------------------------------------

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

// NB: conv_output_size and conv_input_size are not bijections,
// as conv_output_size loses information; this is why conv_input_size
// takes an extra output_padding argument to resolve the ambiguity.


// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

// Note [Legacy CuDNN grouped convolution support]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CuDNN earlier than CuDNN 7 does not directly support group
// convolution, so we provide support for it by sequentially
// running a convolution per group  with appropriately
// adjusted sizes.  https://blog.yani.io/filter-group-tutorial/
// has a fairly good diagram explaining how it works.

// Used on pad, stride and dilation
static void check_args(CheckedFrom c, IntList args, size_t expected_size, const char* arg_name)
{
  if (args.size() > expected_size){
    std::stringstream ss;
    ss << "Too many " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }
  else if (args.size() < expected_size){
    std::stringstream ss;
    ss << "Not enough " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    throw std::runtime_error(ss.str());
  }
}


// NB: For many call sites, it is not strictly necessary to check all of
// these relationships (for example, for forward convolution, we compute
// the size of output ourselves, so we don't actually need to check
// output.  However, writing a single function that does everything
// means we get to reuse it for both forwards and all backwards
// variants, even when the set of "real" inputs varies.  The magic of
// relational computing!
//
// (There is one downside, which is that it is slightly harder to write
// error messages which are able to distinguish between real inputs
// (which the user can change) and computed inputs (which the user can
// only indirectly affect).  It would be an interesting exercise to
// come up with a general framework to handle such situations.)
static void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntList padding, IntList stride, IntList dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams
{
  cudnnDataType_t dataType;
  //int input_size[2 + max_dim];
  //int input_stride[2 + max_dim];
  //int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    cudnnDataType_t dataType,
    IntList padding, IntList stride, IntList dilation,
    int64_t groups, bool deterministic) {

  memset(params, 0, sizeof(ConvolutionParams));
  params->dataType = dataType;
  //// ASSERT(weight.dim() == input.dim())
  //for (int i = 0; i != input.dim(); ++i) {
  //  params->input_size[i] = (int) input.size(i);
  //  params->input_stride[i] = (int) input.stride(i);
  //  params->weight_size[i] = (int) weight.size(i);
  //}
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
  params->deterministic = deterministic;
}

// Convenience struct for passing around descriptors and data
// pointers
struct ConvolutionArgs {
  cudnnHandle_t handle;
  ConvolutionParams params;
  TensorDescriptor idesc, odesc;
  FilterDescriptor wdesc;
  //const Tensor& input, output, weight;
  const void *input, *output, *weight;
  ConvolutionDescriptor cdesc;

  //ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight) : input(input), output(output), weight(weight) {
  ConvolutionArgs(const void *input, const void *output, const void *weight) : input(input), output(output), weight(weight) {
  }
};

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

// TODO: Use something less heavy duty than a big honking mutex
template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<ConvolutionParams, T, ParamsHash<ConvolutionParams>, ParamsEqual<ConvolutionParams>> map;

  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }
};

BenchmarkCache<cudnnConvolutionFwdAlgo_t> fwd_algos;
BenchmarkCache<cudnnConvolutionBwdDataAlgo_t> bwd_data_algos;
BenchmarkCache<cudnnConvolutionBwdFilterAlgo_t> bwd_filter_algos;

// TODO: Stop manually allocating CUDA memory; allocate an ATen byte
// tensor instead.
struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    AT_CUDA_CHECK(THCudaMalloc(globalContext().lazyInitCUDA(), &data, size));
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      THCudaFree(globalContext().lazyInitCUDA(), data);
    }
  }

  size_t size;
  void* data;
};

template<typename algo_t>
struct algorithm_search {
};

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionFwdAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        sz
    );
}
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        sz);
}
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        sz);
}

template<typename algo_t>
size_t getMaxWorkspaceSize(
    const ConvolutionArgs& args,
    const algo_t *algo, int n_algo)
{
    THCState *state = globalContext().lazyInitCUDA();

    size_t max_ws_size = 0;
    size_t max_block_size = 0;
    size_t total_gpu_mem = 0;
    size_t free_gpu_mem = 0;

    THCudaCheck(THCudaMemGetInfoCached(state, &free_gpu_mem, &total_gpu_mem, &max_block_size));

    for (int i = 0; i < n_algo; i++) {
        cudnnStatus_t err;
        size_t sz;
        err = getWorkspaceSize(args, algo[i], &sz);
        if (CUDNN_STATUS_SUCCESS != err || sz == 0
            || sz < max_ws_size || sz > max_block_size) continue;
        max_ws_size = sz;
    }
    return max_ws_size;
}

template<typename perf_t>
perf_t getBestAlgorithm(perf_t *perfResults, bool deterministic, int n_algo) {
  if (deterministic) {
    // iterate over perf results of all algorithms and find the best deterministic algo
    for (int i = 0; i < n_algo; i++) {
      // TODO: Shouldn't all returned results be successful?
      // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
      if (perfResults[i].status == CUDNN_STATUS_SUCCESS &&
          perfResults[i].determinism == CUDNN_DETERMINISTIC) {
        return perfResults[i];
      }
    }
    throw std::runtime_error("no deterministic convolution algorithms available in CuDNN");
  } else {
    return perfResults[0];
  }
}

template<>
struct algorithm_search<cudnnConvolutionFwdAlgo_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static BenchmarkCache<algo_t>& cache() { return fwd_algos; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    static const algo_t algos[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution forward algorithms");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
    Workspace ws(max_ws_size);
    AT_CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        args.handle,
        args.idesc.desc(), args.input,
        args.wdesc.desc(), args.weight,
        args.cdesc.desc(),
        args.odesc.desc(), args.output,
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getBestAlgorithm(perf_results.get(), args.params.deterministic, perf_count);
  }

  static void getAlgorithm(
    const ConvolutionArgs& args,
    algo_t* algo)
  {
    cudnnConvolutionFwdPreference_t pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        pref,
        0,
        algo));
  }

  static void getWorkspaceSize(
    const ConvolutionArgs& args,
    algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        workspaceSize));
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdDataAlgo_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static BenchmarkCache<algo_t>& cache() { return bwd_data_algos; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward data algorithms.");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
    Workspace ws(max_ws_size);
    AT_CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
        args.handle,
        args.wdesc.desc(), args.weight,
        args.odesc.desc(), args.output,
        args.cdesc.desc(),
        args.idesc.desc(), args.input,
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getBestAlgorithm(perf_results.get(), args.params.deterministic, perf_count);
  }

  static void getAlgorithm(const ConvolutionArgs& args, algo_t* algo) {
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        algo));
  }

  static void getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        workspaceSize));
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdFilterAlgo_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static BenchmarkCache<algo_t>& cache() { return bwd_filter_algos; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
#if CUDNN_VERSION >= 6000
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
#endif
    };
    // NOTE: - 1 because ALGO_WINOGRAD is not implemented
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward filter algorithms.");
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
    int perf_count;
    Workspace ws(max_ws_size);

    AT_CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        args.handle,
        args.idesc.desc(), args.input,
        args.odesc.desc(), args.output,
        args.cdesc.desc(),
        args.wdesc.desc(), args.weight,
        num_algos,
        &perf_count,
        perf_results.get(),
        ws.data,
        ws.size));
    return getBestAlgorithm<perf_t>(perf_results.get(), args.params.deterministic, perf_count);
  }

  static void getAlgorithm(const ConvolutionArgs& args, algo_t* algo) {
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        algo)
    );
  }

  static void getWorkspaceSize(const ConvolutionArgs& args, algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        workspaceSize));
  }
};

template<typename algo_t>
void findAlgorithm(const ConvolutionArgs& args, bool benchmark, algo_t* algo) {
  using search = algorithm_search<algo_t>;
  auto& cache = search::cache();

  if (cache.find(args.params, algo)) {
    return;
  }

  if (args.params.deterministic && !benchmark) {
    *algo = search::DEFAULT_ALGO;
    return;
  }

  if (!benchmark) {
    search::getAlgorithm(args, algo);
    return;
  }

  if (cache.find(args.params, algo)) {
    // re-check cache since another thread may have benchmarked the algorithm
    return;
  }

  auto perfResults = search::findAlgorithm(args);
  // for deterministic algo, look at all the perf results and return the best
  // deterministic algo
  if (perfResults.status == CUDNN_STATUS_SUCCESS &&
      !(args.params.deterministic && perfResults.determinism != CUDNN_DETERMINISTIC)) {
      *algo = perfResults.algo;
  } else {
      *algo = search::DEFAULT_ALGO;
  }
  cache.insert(args.params, *algo);

  // These two lines frees the cached blocks in our caching allocator. They are
  // needed here because the above benchmarking uses a huge amount of memory,
  // e.g. a few GBs.
  THCDeviceAllocator* allocator = THCCachingAllocator_get();
  AT_CUDA_CHECK(allocator->emptyCache(allocator->state));
}

template<typename algo_t>
Workspace chooseAlgorithm(
    const ConvolutionArgs& args,
    bool benchmark,
    algo_t* algo)
{
  findAlgorithm(args, benchmark, algo);

  using search = algorithm_search<algo_t>;
  size_t workspace_size;
  search::getWorkspaceSize(args, *algo, &workspace_size);
  try {
    return Workspace(workspace_size);
  } catch (std::runtime_error& e) {
    cudaGetLastError(); // clear OOM error

    // switch to default algorithm and record it in the cache to prevent
    // further OOM errors
    *algo = search::DEFAULT_ALGO;
    search::cache().insert(args.params, *algo);

    search::getWorkspaceSize(args, *algo, &workspace_size);
    return Workspace(workspace_size);
  }
}

// ---------------------------------------------------------------------
//
// Bias addition
//
// ---------------------------------------------------------------------

// In-place!
void raw_cudnn_convolution_add_bias_(
    cudnnDataType_t dataType,
    void *output, const void *bias,
    IntList oshape, IntList bshape,
    IntList ostride, IntList bstride)
{
  // bshape = 1s
  // bshape[axis] = oshape[axis]

  // See Note [CuDNN broadcast padding].  Handle the left padding
  // ourselves, but use TensorDescriptor's padding argument to do the rest.
  TensorDescriptor bdesc, odesc;

  bdesc.set(dataType, bshape, bstride);
  odesc.set(dataType, oshape, ostride);

  auto handle = getCudnnHandle();
  Constant one(dataType, 1);

  AT_CUDNN_CHECK(cudnnAddTensor(handle, &one, bdesc.desc(), bias,
                                     &one, odesc.desc(), output));
}

// The general strategy:
//
//    - cudnn_convolution (Tensor)
//      Entry points for clients, takes bias
//
//    - cudnn_convolution_forward (TensorArg)
//      Entry point, which may be reused between regular
//      convolution and transposed convolution.  Does NOT take bias.
//
//    - raw_cudnn_convolution_forward_out (Tensor)
//      Low level function which invokes CuDNN, and takes an output
//      tensor which is directly written to (thus _out).
//
// Where does argument checking happen?  Here's the division of
// responsibility:
//  - Things that happen in at::Tensor
//    - TensorArg allocation
//    - setCuDNNStreamToCurrent
//  - Things that happen in TensorArg
//    - Check arguments (type, GPU, shape)
//
// TODO: Consider renaming zero-indexed arguments to "self"



// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// The raw API directly invokes CuDNN and does not emulate support
// for group convolution on old versions of CuDNN.
//
// There are a few reasons this should never be directly exposed
// via ATen:
//
//    - It takes output as a parameter (this should be computed!)
//    - It doesn't do input checking
//    - It doesn't resize output (it is assumed to be correctly sized)
//    - It takes a ConvolutionParams struct
//

void raw_cudnn_convolution_forward_out(
    cudnnDataType_t dataType,
    void* output, const void* input, const void* weight,
    IntList oshape, IntList ishape, IntList wshape,
    IntList istride, IntList ostride,
    IntList padding, IntList stride, IntList dilation, int64_t groups,
    int axis, bool benchmark, bool deterministic) {

//void raw_cudnn_convolution_forward_out(
//    const Tensor& output, const Tensor& input, const Tensor& weight,
//    IntList padding, IntList stride, IntList dilation, int64_t groups,
//    bool benchmark, bool deterministic) {

  ConvolutionArgs args{ input, output, weight }; //can pass pointer
  args.handle = getCudnnHandle();

  // needs input, actually we don't need input/weight stride/shape for params!
  setConvolutionParams(&args.params, dataType, padding, stride, dilation, groups, deterministic);
  args.idesc.set(dataType, ishape, istride);
  args.wdesc.set(dataType, wshape, axis);
  args.odesc.set(dataType, oshape, ostride);
  args.cdesc.set(dataType, wshape.size() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  // TODO: when we do legacy group convolution support, we'll repeatedly
  // reinitialize the workspace for each convolution we do.  This is
  // wasteful; we'd rather reuse the workspace.  OTOH, legacy group
  // convolution support is already pretty slow, so this might not
  // matter.  (This applies to raw_cudnn_convolution_backward_input as well.)
  cudnnConvolutionFwdAlgo_t fwdAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionForward(
    args.handle,
    &one, args.idesc.desc(), input,
    args.wdesc.desc(), weight,
    args.cdesc.desc(), fwdAlg, workspace.data, workspace.size,
    &zero, args.odesc.desc(), output));
}


// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------


void raw_cudnn_convolution_backward_input_out(
    cudnnDataType_t dataType,
    void* grad_input, const void* grad_output, const void* weight,
    IntList ishape, IntList oshape, IntList wshape,
    IntList istride, IntList ostride,
    IntList padding, IntList stride, IntList dilation, int64_t groups,
    int axis, bool benchmark, bool deterministic) {

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, dataType, padding, stride, dilation, groups, deterministic);
  args.idesc.set(dataType, ishape, istride);
  args.wdesc.set(dataType, wshape, 0, axis);
  args.odesc.set(dataType, oshape, ostride);
  args.cdesc.set(dataType, wshape.size() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  cudnnConvolutionBwdDataAlgo_t bwdDataAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardData(
      args.handle,
      &one, args.wdesc.desc(), weight,
      args.odesc.desc(), grad_output,
      args.cdesc.desc(), bwdDataAlg, workspace.data, workspace.size,
      &zero, args.idesc.desc(), grad_input));
}


// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_weight_out(
    cudnnDataType_t dataType,
    void *grad_weight, const void *grad_output, const void *input,
    IntList ishape, IntList oshape, IntList wshape,
    IntList istride, IntList ostride,
    IntList padding, IntList stride, IntList dilation, int64_t groups,
    int axis, bool benchmark, bool deterministic) {

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, dataType, padding, stride, dilation, groups, deterministic);
  args.idesc.set(dataType, ishape, istride);
  args.wdesc.set(dataType, wshape, 0, axis);
  args.odesc.set(dataType, oshape, ostride);
  args.cdesc.set(dataType, wshape.size() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
      args.handle,
      &one, args.idesc.desc(), input,
      args.odesc.desc(), grad_output,
      args.cdesc.desc(), bwdFilterAlg, workspace.data, workspace.size,
      &zero, args.wdesc.desc(), grad_weight));
}


// ---------------------------------------------------------------------
//
// Convolution backward (bias)
//
// ---------------------------------------------------------------------


void raw_cudnn_convolution_backward_bias(
    cudnnDataType_t dataType,
    void *grad_bias, const void *grad_output,
    IntList oshape, IntList bshape,
    IntList ostride, IntList bstride)
{
  setCuDNNStreamToCurrent();

  // See Note [CuDNN broadcast padding].  Handle the left padding
  // ourselves, but use TensorDescriptor's pad argument to do the rest.
  TensorDescriptor bdesc, odesc;
  bdesc.set(dataType, bshape, bstride);
  odesc.set(dataType, oshape, ostride);

  auto handle = getCudnnHandle();
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  AT_CUDNN_CHECK(cudnnConvolutionBackwardBias(handle, &one, odesc.desc(), grad_output,
                                                   &zero, bdesc.desc(), grad_bias));
}

}}  // namespace

#endif
