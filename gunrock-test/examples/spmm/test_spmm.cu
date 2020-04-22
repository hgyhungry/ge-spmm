// ----------------------------------------------------------------
#include <gunrock/app/spmm/spmm_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

/******************************************************************************
 * Main
 ******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct {
  /**
   * @brief the actual main function, after type switching
   * @tparam VertexT    Type of vertex identifier
   * @tparam SizeT      Type of graph size, i.e. type of edge identifier
   * @tparam ValueT     Type of edge values
   * @param  parameters Command line parameters
   * @param  v,s,val    Place holders for type deduction
   * \return cudaError_t error message(s), if any
   */
  template <typename VertexT,  // Use int as the vertex identifier
            typename SizeT,    // Use int as the graph size type
            typename ValueT>   // Use int as the value type
  cudaError_t
  operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
    // CLI parameters
    bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");

    typedef typename app::TestGraph<VertexT, SizeT, ValueT,
                                    graph::HAS_EDGE_VALUES | graph::HAS_COO |
                                        graph::HAS_CSR | graph::HAS_CSC>
        GraphT;

    cudaError_t retval = cudaSuccess;
    util::CpuTimer cpu_timer;
    GraphT graph;

    cpu_timer.Start();
    GUARD_CU(graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    int feature_len = parameters.Get<int>("feature-len");
    ValueT *h_output = new ValueT[(graph.nodes*feature_len)];
    ValueT *ref_output;
    
    if (!quick) {
      // ref_output = new ValueT[graph.nodes*feature_len];
      
      // util::PrintMsg("__________________________", !quiet);

      // float elapsed =
      //     app::spmm::CPU_Reference(parameters, graph, h_input, ref_output, quiet);

      // util::PrintMsg(
      //     "--------------------------\n Elapsed: " + std::to_string(elapsed),
      //     !quiet);
    }

    // <TODO> add other switching parameters, if needed
    std::vector<std::string> switches{"advance-mode"};
    // </TODO>

    GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
                                    [
                                        // </TODO> pass necessary data to lambda
                                        // ref_degrees
                                        // </TODO>
    ](util::Parameters &parameters, GraphT &graph) {
                                      // <TODO> pass necessary data to
                                      // app::Template::RunTests
                                      return app::spmm::RunTests(
                                          parameters, graph
                                          // util::DEVICE
                                          );
                                      // </TODO>
                                    }));

    if (!quick) {
      // delete[] ref_output;
      // ref_output = NULL;
      
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test spmm");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::spmm::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  // TODO: change available graph types, according to requirements
  return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B | app::SIZET_U64B |
                           app::VALUET_F32B | app::DIRECTED | app::UNDIRECTED>(
      parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
