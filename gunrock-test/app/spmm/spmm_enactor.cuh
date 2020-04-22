#pragma once
#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/util/track_utils.cuh>
#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/spmm/spmm_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace spmm {

/**
 * @brief Speciflying parameters for spmm Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

/**
 * @brief defination of spmm iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct SPMMIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push 
                                            //  |
                                            //  (((EnactorT::Problem::FLAG &
                                            //     Mark_Predecessors) != 0)
                                            //     ? Update_Predecessors
                                            //     : 0x0)
                                            > {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Push 
                                          //   |
                                          // (((EnactorT::Problem::FLAG &
                                          //    Mark_Predecessors) != 0)
                                          //      ? Update_Predecessors
                                          //      : 0x0)
                                               >
      BaseIterationLoop;

  SPMMIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of spmm, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data spmm that works on
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    // auto         &row_offsets        =   graph.CsrT::row_offsets;
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &stream = enactor_slice.stream;
    // auto         &stream             =   enactor_slice.stream;
    auto &iteration = enactor_stats.iteration;
    auto &input = data_slice.input;
    auto &output = data_slice.output;
    auto &feature_len = data_slice.feature_len;
    
    auto &vertices = data_slice.vertices;
    auto null_ptr = &vertices;
    null_ptr = NULL;
    // util::Array1D<SizeT, VertexT> *null_frontier = NULL;
    util::PrintMsg("nodes." + std::to_string(graph.nodes), 1);

    long ct = 0;
    util::PrintMsg("counts." + std::to_string(ct), 1);


    // (h1 + h2 + h3) edge map
    frontier.queue_length = vertices.GetSize();
    frontier.queue_reset = true;
    oprtr_parameters.advance_mode = "ALL_EDGES";
    auto advance_op1 =
    [feature_len, input, output] __host__ __device__(
      const VertexT &src, VertexT &dest, const SizeT &edge_id,
      const VertexT &input_item, const SizeT &input_pos,
      SizeT &output_pos) -> bool {
        for (int j = 0; j < feature_len; j++) {
          atomicAdd(output+ dest * feature_len + j, input[src * feature_len + j]);
        }
        // printf(" src %d dest %d \n",src,dest);
        return true; 
      };
    GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
      graph.csr(), &vertices, null_ptr, oprtr_parameters,
      advance_op1));
   
    util::PrintMsg("core end");

    GUARD_CU2(cudaMemcpyAsync(
              data_slice.h_output.GetPointer(util::HOST),
              output.GetPointer(util::DEVICE),
              ((uint64_t)graph.nodes) * feature_len * sizeof(ValueT),
              cudaMemcpyDeviceToHost, data_slice.d2h_stream),
          "source_result D2H copy failed");
    util::PrintMsg("copy end");

    return retval;
  }

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each
   * transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each
   * transmition item, typed ValueT
   * @param  received_length The numver of transmition items received
   * @param[in] peer_ which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    util::PrintMsg("ExpandIncoming", 1);
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    
    auto expand_op = [] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
      return true;
    };

  

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    int num_gpus = this->enactor->num_gpus;
    auto &enactor_slices = this->enactor->enactor_slices;

    for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++) {
      auto &retval = enactor_slices[gpu].enactor_stats.retval;
      if (retval == cudaSuccess) continue;
      printf("(CUDA error %d @ GPU %d: %s\n", retval, gpu % num_gpus,
             cudaGetErrorString(retval));
      fflush(stdout);
      return true;
    }

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor->enactor_slices[this->gpu_num * this->enactor->num_gpus];
    
    if (enactor_slice.enactor_stats.iteration > 0) return true;

    return false;
  }

  cudaError_t Compute_OutputLength(int peer_) { return cudaSuccess; }

  cudaError_t Check_Queue_Size(int peer_) { return cudaSuccess; }

};  // end of SPMMIteration

/**
 * @brief spmm enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<typename _Problem::GraphT, typename _Problem::LabelT,
                         typename _Problem::ValueT, ARRAY_FLAG,
                         cudaHostRegisterFlag> {
 public:
  // Definations
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::GraphT GraphT;
  typedef typename Problem::LabelT LabelT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef SPMMIterationLoop<EnactorT> IterationT;

  // Members
  Problem *problem;
  IterationT *iterations;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief SPMMEnactor constructor
   */
  Enactor() : BaseEnactor("spmm"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief SPMMEnactor destructor
   */
  virtual ~Enactor() {
    // Release();
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Release(target));
    delete[] iterations;
    iterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
   * @brief Initialize the enactor.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 0, NULL, target, false));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));
    }

    iterations = new IterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          
        }
      }
    }
    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief one run of spmm, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    util::PrintMsg("spmm enactor Enact run.", 1);
    gunrock::app::Iteration_Loop<0, 1,
        IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a spmm computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    util::PrintMsg("spmm enactor Enact.", 1);
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU spmm Done.", this->flag & Debug);
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
