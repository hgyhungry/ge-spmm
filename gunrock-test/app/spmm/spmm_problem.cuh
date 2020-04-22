
#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace spmm {

/**
 * @brief Speciflying parameters for SPMM Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));
  // GUARD_CU(parameters.Use<bool>(
  //     "mark-pred",
  //     util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
  //     false, "Whether to mark predecessor info.", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Single-Source Shortest Path Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in SPMM
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _LabelT = typename _GraphT::VertexT,
          typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::CooT CooT;
  typedef typename GraphT::GpT GpT;
  typedef _LabelT LabelT;
  typedef _ValueT ValueT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // Helper structures

  /**
   * @brief Data structure containing SPMM-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // SPMM-specific storage arrays
    util::Array1D<SizeT, ValueT> input;
    util::Array1D<SizeT, ValueT> output;
    util::Array1D<SizeT, ValueT> h_output;
    util::Array1D<SizeT, VertexT> vertices;  
    cudaStream_t d2h_stream;

    int feature_len;
    // long long counts;
    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      input.SetName("input");
      output.SetName("output");
      // tmp.SetName("tmp");
      // H2.SetName("H2");
      h_output.SetName("h_output");
      // W1.SetName("W1");
      // W2.SetName("W2");
      // feature_len.SetName("feature_len");
      // degrees.SetName("degrees");
      vertices.SetName("vertices");
      // preds.SetName("preds");
      // temp_preds.SetName("temp_preds");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      util::PrintMsg("spmm probem Release", 1);
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      // GUARD_CU(input.Release(target));
      // GUARD_CU(H1.Release(target));
      // GUARD_CU(H2.Release(target));
      // GUARD_CU(tmp.Release(target));
      // GUARD_CU(host_H2.Release(util::HOST));
      // GUARD_CU(W1.Release(target));
      // GUARD_CU(W2.Release(target));
      // GUARD_CU(degrees.Release(target));
      // GUARD_CU(vertices.Release(target));
      GUARD_CU(BaseDataSlice ::Release(target));
      GUARD_CU2(cudaStreamDestroy(d2h_stream), "cudaStreamDestory failed.");
      return retval;
    }

    /**
     * @brief initializing SPMM-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] num_gpus    Number of GPUs
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;
      util::PrintMsg("SPMM probem init", 1);

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
      GUARD_CU(input.Allocate(sub_graph.nodes * feature_len, target));
      GUARD_CU(vertices.Allocate(sub_graph.nodes, target));
      GUARD_CU(output.Allocate(sub_graph.nodes * feature_len, target));
      GUARD_CU(h_output.Allocate(sub_graph.nodes * feature_len, util::HOST));
      // GUARD_CU(tmp.Allocate(sub_graph.nodes * feature_size, target));
      // GUARD_CU(H1.Allocate(sub_graph.nodes * feature_size, target));
      // GUARD_CU(H2.Allocate(sub_graph.nodes * feature_size, target));
      // GUARD_CU(host_H2.Allocate(sub_graph.nodes * feature_size, util::HOST));
      // GUARD_CU(W1.Allocate(feature_size * feature_size, target));
      // GUARD_CU(W2.Allocate(feature_size * feature_size, target));
      // GUARD_CU(degrees.Allocate(sub_graph.nodes + 1, target));
      // if (flag & Mark_Predecessors) {
      //   GUARD_CU(preds.Allocate(sub_graph.nodes, target));
      //   GUARD_CU(temp_preds.Allocate(sub_graph.nodes, target));
      // }
      GUARD_CU2(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags failed.");
      GUARD_CU(vertices.ForAll(
          [] __host__ __device__(VertexT * l_vertices, const SizeT &pos) {
            l_vertices[pos] = pos;
          },
          sub_graph.nodes, target, this->stream));

      /*if (target & util::DEVICE)
      {
          GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this -> stream));
      }*/
      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      util::PrintMsg("spmm probem Reset", 1);
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;

      // // Ensure data are allocated
      // GUARD_CU(input.EnsureSize_(nodes * feature_size, target));
      // GUARD_CU(H1.EnsureSize_(nodes * feature_size, target));
      // GUARD_CU(H2.EnsureSize_(nodes * feature_size, target));
      // GUARD_CU(W1.EnsureSize_(feature_size * feature_size, target));
      // GUARD_CU(W2.EnsureSize_(feature_size * feature_size, target));
      // if (this->flag & Mark_Predecessors) {
      //   GUARD_CU(preds.EnsureSize_(nodes, target));
      //   GUARD_CU(temp_preds.EnsureSize_(nodes, target));
      // }

      // Reset data
      GUARD_CU(input.ForEach(
          [] __host__ __device__(ValueT & val) { val = 1; },
          util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));
      GUARD_CU(output.ForEach(
          [] __host__ __device__(ValueT & val) { val = 0; },
          util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));
        // GUARD_CU(W1.ForEach(
      //     [] __host__ __device__(ValueT & val) { val = 0; },
      //     util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));
      // GUARD_CU(H2.ForEach(
      //     [] __host__ __device__(ValueT & val) { val = 0; },
      //     util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));
      // GUARD_CU(W2.ForEach(
      //     [] __host__ __device__(ValueT & val) { val = 0; },
      //     util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));
      // GUARD_CU(vertices.ForEach(
      //     [] __host__ __device__(ValueT & val) { val = 0; },
      //     util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));
      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief SPMMProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief SPMMProblem default destructor
   */
  virtual ~Problem() { Release(); }

  /*
   * @brief Releasing allocated memory space
   * @param[in] target      The location to release memory from
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    util::PrintMsg("spmm probem Release", 1);
    cudaError_t retval = cudaSuccess;
    if (data_slices == NULL) return retval;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i].Release(target));

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Copy result distancess computed on GPUs back to host-side arrays.
   * @param[out] h_distances Host array to store computed vertex distances from
   * the source.
   * @param[out] h_preds     Host array to store computed vertex predecessors.
   * @param[in]  target where the results are stored
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(ValueT *h_output, 
                      // VertexT *h_preds = NULL,
                      util::Location target = util::DEVICE) {
      util::PrintMsg("spmm probem Extract", 1);
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;
    auto &data_slice = data_slices[0][0];
    GUARD_CU(data_slice.h_output.ForEach(
        h_output,
        [] __host__ __device__(const ValueT &val, ValueT &h_val) {
          h_val = val;
        },
        ((uint64_t)nodes) * data_slice.feature_len, util::HOST));
    // Set device
    util::PrintMsg("spmm probem Extract over", 1);
    return retval;
  }


  // template <typename ArrayT>
  // cudaError_t ReadMat(ArrayT &array, std::string filename, uint64_t dim0,
  //                     uint64_t dim1) {
  //   cudaError_t retval = cudaSuccess;

  //   auto temp_vals_2D = gunrock::app::sage::template ReadMatrix<ValueT, SizeT>(
  //       filename, dim0, dim1);
  //   GUARD_CU(array.Allocate(dim0 * dim1, util::HOST));
  //   // for (auto pos = 0; pos < dim0 * dim1; pos++)
  //   //{
  //   //    array[pos] = temp_vals_2D[pos / dim1][pos % dim1];
  //   //}
  //   GUARD_CU(array.ForAll(
  //       [temp_vals_2D, dim1] __host__ __device__(ValueT * vals,
  //                                                const uint64_t &pos) {
  //         vals[pos] = temp_vals_2D[pos / dim1][pos % dim1];
  //       },
  //       dim0 * dim1, util::HOST));
  //   for (auto x = 0; x < dim0; x++) {
  //     delete[] temp_vals_2D[x];
  //     temp_vals_2D[x] = NULL;
  //   }
  //   delete[] temp_vals_2D;
  //   temp_vals_2D = NULL;

  //   return retval;
  // }


  /**
   * @brief initialization function.
   * @param     graph       The graph that SPMM processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
      util::PrintMsg("spmm probem Init2", 1);
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];
    auto &para = this->parameters;

    // if (this->parameters.template Get<bool>("mark-pred"))
    //   this->flag = this->flag | Mark_Predecessors;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      // data_slice.counts = 0;
      data_slice.feature_len = para.template Get<int>("feature-len");
      printf("feature len %d", data_slice.feature_len);
      // ??
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));
    }  // end for (gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(//VertexT src,
   util::Location target = util::DEVICE) {
      util::PrintMsg("spmm probem reset2", 1);
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));

      // int rand_seed = this->parameters.template Get<int>("rand-seed");
      // if (!util::isValid(rand_seed)) rand_seed = time(NULL);
      // if (!this->parameters.template Get<bool>("quiet"))
      //   util::PrintMsg("rand-seed = " + std::to_string(rand_seed));

      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

      return retval;
    }

  /** @} */
};

}  // namespace spmm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
