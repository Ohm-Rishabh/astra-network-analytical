/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Helper.h"
#include "congestion_aware/FullyConnected.h"
#include "congestion_aware/Ring.h"
#include "congestion_aware/Switch.h"
#include "congestion_aware/Mesh2D.h"
#include "congestion_aware/SparseMesh2D.h"
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> NetworkAnalyticalCongestionAware::construct_topology(
    const NetworkParser& network_parser) noexcept {
    // get network_parser info
    const auto dims_count = network_parser.get_dims_count();
    const auto topologies_per_dim = network_parser.get_topologies_per_dim();
    const auto npus_counts_per_dim = network_parser.get_npus_counts_per_dim();
    const auto bandwidths_per_dim = network_parser.get_bandwidths_per_dim();
    const auto latencies_per_dim = network_parser.get_latencies_per_dim();

    // for now, congestion_aware backend supports 1-dim topology only
    if (dims_count != 1) {
        std::cerr << "[Error] (network/analytical/congestion_aware) " << "only support 1-dim topology" << std::endl;
        std::exit(-1);
    }

    // retrieve basic basic-topology info
    const auto topology_type = topologies_per_dim[0];
    const auto npus_count = npus_counts_per_dim[0];
    const auto bandwidth = bandwidths_per_dim[0];
    const auto latency = latencies_per_dim[0];

    // get mesh dimensions (for Mesh2D topology)
    const auto mesh_width = network_parser.get_mesh_width();
    const auto mesh_height = network_parser.get_mesh_height();
    
    // get excluded coordinates (for SparseMesh2D topology)
    const auto excluded_coords = network_parser.get_excluded_coords();
    
    // get custom NPU placement (for SparseMesh2D topology with custom layout)
    const auto npu_placement = network_parser.get_npu_placement();

    switch (topology_type) {
    case TopologyBuildingBlock::Ring:
        return std::make_shared<Ring>(npus_count, bandwidth, latency);
    case TopologyBuildingBlock::Switch:
        return std::make_shared<Switch>(npus_count, bandwidth, latency);
    case TopologyBuildingBlock::FullyConnected:
        return std::make_shared<FullyConnected>(npus_count, bandwidth, latency);
    case TopologyBuildingBlock::Mesh2D:
        // Use explicit width/height if provided, otherwise fall back to square mesh
        if (mesh_width > 0 && mesh_height > 0) {
            return std::make_shared<Mesh2D>(mesh_width, mesh_height, bandwidth, latency);
        } else {
            return std::make_shared<Mesh2D>(npus_count, bandwidth, latency);
        }
    case TopologyBuildingBlock::SparseMesh2D:
        // SparseMesh2D requires width, height, and excluded coordinates
        if (mesh_width > 0 && mesh_height > 0) {
            // Use custom placement constructor if npu_placement is provided
            if (!npu_placement.empty()) {
                return std::make_shared<SparseMesh2D>(mesh_width, mesh_height, excluded_coords, 
                                                      npu_placement, bandwidth, latency);
            } else {
                return std::make_shared<SparseMesh2D>(mesh_width, mesh_height, excluded_coords, 
                                                      bandwidth, latency);
            }
        } else {
            std::cerr << "[Error] (network/analytical/congestion_aware) SparseMesh2D requires width and height" << std::endl;
            std::exit(-1);
        }
    default:
        // shouldn't reaach here
        std::cerr << "[Error] (network/analytical/congestion_aware) " << "not supported basic-topology" << std::endl;
        std::exit(-1);
    }
}
