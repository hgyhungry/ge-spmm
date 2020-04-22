#pragma once

namespace gunrock {
namespace app {
namespace spmm {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the spmm result (i.e., distance from source)
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] preds Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template <typename T, typename SizeT>
void DisplaySolution(T *array, SizeT length) {
  if (length > 40) length = 40;

  util::PrintMsg("[", true, false);
  for (SizeT i = 0; i < length; ++i) {
    util::PrintMsg(std::to_string(i) + ":" + std::to_string(array[i]) + " ",
                   true, false);
  }
  util::PrintMsg("]");
}

/******************************************************************************
 * spmm Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference spmm implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the spmm
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double CPU_Reference(util::Parameters &para, const GraphT &graph, 
                      ValueT *input, ValueT *output, bool quiet) 
{
  
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  util::PrintMsg("CPU Reference\n");
  int feature_len = para.template Get<int>("feature-len");
  bool debug = para.template Get<bool>("v");
  
  typedef std::pair<VertexT, VertexT> EdgeT;

  EdgeT *edges = (EdgeT *)malloc(sizeof(EdgeT) * graph.edges);
  
  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  for (int v=0; v<graph.nodes; v++) {
    for (int f=0; f<feature_len; f++) {
      output[(feature_len*v+f)] = 0;
    }
  }

  for (VertexT v = 0; v < graph.nodes; ++v) {
    SizeT st = graph.CsrT::GetNeighborListOffset(v);
    SizeT len = graph.CsrT::GetNeighborListLength(v);
    for (SizeT ptr=st; ptr<st+len; ptr++) {
      SizeT u = graph.CsrT::GetEdgeDest(ptr);
      for (SizeT f=0; f<feature_len; f++)
      output[(feature_len*u+f)] += input[(feature_len*v+f)];
    }
  }
  
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  // // util::PrintMsg("CPU spmm finished in " + std::to_string(elapsed)
  // //    + " msec.", !quiet);

  return elapsed;
}

/**
 * @brief Validation of spmm results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  src           The source vertex
 * @param[in]  h_distances   Computed distances from the source to each vertex
 * @param[in]  h_preds       Computed predecessors for each vertex
 * @param[in]  ref_distances Reference distances from the source to each vertex
 * @param[in]  ref_preds     Reference predecessors for each vertex
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(util::Parameters &parameters,
                                        GraphT &graph, ValueT *h_output,
                                        uint64_t feature_len,
                                        bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");

  util::PrintMsg("spmm validation: ");
  // Verify the result

  // compute reference
  ValueT *ref_output = new ValueT[((uint64_t)graph.nodes)];
  for (SizeT v = 0; v < graph.nodes; v++) { ref_output = 0;}
  for (SizeT v = 0; v < graph.nodes; v++) {
    SizeT st = graph.CsrT::GetNeighborListOffset(v);
    for (SizeT e = 0; e < graph.CsrT::GetNeighborListLength(v); e++) {
      SizeT u = graph.CsrT::GetEdgeDest(st+e);
      ref_output[u] += 1;
    }
  }

  util::PrintMsg("ref calc over ");

  for (SizeT v = 0; v < graph.nodes; v++) {
    for (SizeT f=0; f<feature_len; f++) {
      if (abs(h_output[(feature_len*v+f)] - ref_output[v]) > 1e-4) {
        if (num_errors == 0) {
          util::PrintMsg("Validation Fail", !quiet);
          util::PrintMsg("FAIL. " + std::to_string(v) +
                              "]) = " + std::to_string(h_output[(feature_len*v+f)]) + ", should be " + std::to_string(ref_output[v] ),
                          !quiet);
        }
        num_errors += 1;
      }
    }
    // double L2_vec = 0.0;
    // uint64_t offset = v * result_column;
    // for (uint64_t j = 0; j < result_column; j++) {
    //   L2_vec += embed_result[offset + j] * embed_result[offset + j];
    // }
    // if (abs(L2_vec - 1.0) > 0.000001) {
    //   if (num_errors == 0) {
    //     util::PrintMsg("FAIL. L2(embedding[" + std::to_string(v) +
    //                        "]) = " + std::to_string(L2_vec) + ", should be 1",
    //                    !quiet);
    //   }
    //   num_errors += 1;
    // }
  }

  if (num_errors == 0)
    util::PrintMsg("PASS", !quiet);
  else
    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);

  return num_errors;
}

}  // namespace spmm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
