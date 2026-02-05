/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/SparseMesh2D.h"
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>  // for std::reverse

using namespace NetworkAnalyticalCongestionAware;

// Helper function to calculate valid NPU count before calling base constructor
static int calculate_valid_npu_count(int width, int height, 
                                      const std::set<std::pair<int, int>>& excluded) {
    int total_positions = width * height;
    int excluded_count = 0;
    for (const auto& [x, y] : excluded) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            excluded_count++;
        }
    }
    return total_positions - excluded_count;
}

/**
 * Constructor: Initialize Sparse 2D Mesh Topology
 *
 * This constructor creates a 2D mesh grid with some positions excluded (holes).
 * Valid nodes are numbered contiguously from 0 to (valid_count - 1).
 * Connections are created only between adjacent valid nodes.
 */
SparseMesh2D::SparseMesh2D(const int width, const int height,
                           const std::set<std::pair<int, int>>& excluded_coords,
                           const Bandwidth bandwidth, const Latency latency) noexcept
    : width(width), height(height), excluded(excluded_coords),
      BasicTopology(calculate_valid_npu_count(width, height, excluded_coords),
                    calculate_valid_npu_count(width, height, excluded_coords), 
                    bandwidth, latency) {
    
    // Validate input parameters
    assert(width > 0);
    assert(height > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // =====================================================
    // SPARSE MESH2D TOPOLOGY CONSTRUCTION - DETAILED LOGGING
    // =====================================================
    std::cerr << "\n";
    std::cerr << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║              SPARSE MESH2D TOPOLOGY INITIALIZATION                         ║\n";
    std::cerr << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    
    std::cerr << "\n[SPARSE-MESH2D-INIT] Maximum Grid: " << width << " (width) × " << height << " (height)\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Excluded positions: " << excluded_coords.size() << "\n";
    
    // Print excluded coordinates
    if (!excluded_coords.empty()) {
        std::cerr << "[SPARSE-MESH2D-INIT] Excluded coordinates: ";
        for (const auto& [x, y] : excluded_coords) {
            std::cerr << "(" << x << "," << y << ") ";
        }
        std::cerr << "\n";
    }

    // Initialize grid_to_npu mapping
    grid_to_npu.resize(width * height, -1);
    
    // Count valid positions and assign NPU IDs
    valid_npu_count = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (excluded.find({x, y}) == excluded.end()) {
                // This position is valid
                int grid_idx = coords_to_grid_index(x, y);
                grid_to_npu[grid_idx] = valid_npu_count;
                npu_to_grid.push_back({x, y});
                valid_npu_count++;
            }
        }
    }

    std::cerr << "[SPARSE-MESH2D-INIT] Valid NPUs: " << valid_npu_count << "\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Bandwidth per link: " << bandwidth << " GB/s\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Latency per link: " << latency << " ns\n";

    // Note: npus_count, devices_count, and devices are already set correctly 
    // by BasicTopology constructor (we passed valid_npu_count to it)

    // Fix topology metadata
    dims_count = 2;
    npus_count_per_dim.clear();
    npus_count_per_dim.push_back(width);   // X dimension (max)
    npus_count_per_dim.push_back(height);  // Y dimension (max)
    bandwidth_per_dim.clear();
    bandwidth_per_dim.push_back(bandwidth);
    bandwidth_per_dim.push_back(bandwidth);

    basic_topology_type = TopologyBuildingBlock::Mesh2D;  // Use Mesh2D type for compatibility

    // Print grid layout with NPU IDs
    std::cerr << "\n[SPARSE-MESH2D-INIT] Grid layout:\n\n";
    
    for (int y = 0; y < height; ++y) {
        std::cerr << "                  ";
        for (int x = 0; x < width; ++x) {
            int npu_id = get_npu_at(x, y);
            if (npu_id >= 0) {
                std::cerr << std::setw(3) << npu_id;
            } else {
                std::cerr << "  x";
            }
            
            // Only print horizontal connector if BOTH current AND next position are valid
            if (x < width - 1) {
                int next_npu = get_npu_at(x + 1, y);
                if (npu_id >= 0 && next_npu >= 0) {
                    std::cerr << " --- ";
                } else {
                    std::cerr << "     ";
                }
            }
        }
        std::cerr << "\n";
        
        // Print vertical connectors row
        if (y < height - 1) {
            std::cerr << "                  ";
            for (int x = 0; x < width; ++x) {
                int npu_id = get_npu_at(x, y);
                int below_npu = get_npu_at(x, y + 1);
                if (npu_id >= 0 && below_npu >= 0) {
                    std::cerr << "  |  ";
                } else {
                    std::cerr << "     ";
                }
                if (x < width - 1) std::cerr << "     ";
            }
            std::cerr << "\n";
        }
    }
    std::cerr << "\n";

    // Create mesh links between adjacent valid nodes
    int link_count = 0;
    std::cerr << "[SPARSE-MESH2D-INIT] Creating mesh links...\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int current_npu = get_npu_at(x, y);
            if (current_npu < 0) continue;  // Skip excluded positions

            // Connect to right neighbor (x + 1, y) if valid
            if (x + 1 < width) {
                int right_npu = get_npu_at(x + 1, y);
                if (right_npu >= 0) {
                    connect(current_npu, right_npu, bandwidth, latency, true);
                    link_count += 2;
                    std::cerr << "[SPARSE-MESH2D-LINK] NPU " << current_npu << " (at " << x << "," << y << ") "
                              << "↔ NPU " << right_npu << " (at " << (x+1) << "," << y << ")\n";
                }
            }

            // Connect to bottom neighbor (x, y + 1) if valid
            if (y + 1 < height) {
                int bottom_npu = get_npu_at(x, y + 1);
                if (bottom_npu >= 0) {
                    connect(current_npu, bottom_npu, bandwidth, latency, true);
                    link_count += 2;
                    std::cerr << "[SPARSE-MESH2D-LINK] NPU " << current_npu << " (at " << x << "," << y << ") "
                              << "↔ NPU " << bottom_npu << " (at " << x << "," << (y+1) << ")\n";
                }
            }
        }
    }
    
    std::cerr << "\n[SPARSE-MESH2D-INIT] Total bidirectional links created: " << (link_count / 2) << "\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Total directed links: " << link_count << "\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Sparse Mesh2D topology construction COMPLETE ✓\n";
    std::cerr << "\n";

    // Print ring order for collective communication
    std::cerr << "[SPARSE-MESH2D-INIT] Ring order for collectives: ";
    for (int i = 0; i < valid_npu_count; ++i) {
        std::cerr << i;
        if (i < valid_npu_count - 1) std::cerr << " → ";
    }
    std::cerr << " → 0 (wrap)\n\n";
}

/**
 * Constructor with CUSTOM NPU PLACEMENT
 * 
 * This constructor allows specifying exactly which NPU ID goes at each grid position.
 * This enables optimized layouts like snake patterns where ring neighbors are physically close.
 */
SparseMesh2D::SparseMesh2D(const int width, const int height,
                           const std::set<std::pair<int, int>>& excluded_coords,
                           const std::map<std::pair<int, int>, int>& npu_placement,
                           const Bandwidth bandwidth, const Latency latency) noexcept
    : width(width), height(height), excluded(excluded_coords),
      BasicTopology(calculate_valid_npu_count(width, height, excluded_coords),
                    calculate_valid_npu_count(width, height, excluded_coords), 
                    bandwidth, latency) {
    
    // Validate input parameters
    assert(width > 0);
    assert(height > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    std::cerr << "\n";
    std::cerr << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║         SPARSE MESH2D TOPOLOGY - CUSTOM NPU PLACEMENT                      ║\n";
    std::cerr << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    
    std::cerr << "\n[SPARSE-MESH2D-INIT] Maximum Grid: " << width << " (width) × " << height << " (height)\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Excluded positions: " << excluded_coords.size() << "\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Custom NPU placement: ENABLED\n";
    
    // Print excluded coordinates
    if (!excluded_coords.empty()) {
        std::cerr << "[SPARSE-MESH2D-INIT] Excluded coordinates: ";
        for (const auto& [x, y] : excluded_coords) {
            std::cerr << "(" << x << "," << y << ") ";
        }
        std::cerr << "\n";
    }

    // Calculate expected valid NPU count
    int expected_count = calculate_valid_npu_count(width, height, excluded_coords);
    valid_npu_count = expected_count;

    // Initialize grid_to_npu mapping
    grid_to_npu.resize(width * height, -1);
    npu_to_grid.resize(valid_npu_count);

    // Use custom placement to set up mappings
    std::cerr << "[SPARSE-MESH2D-INIT] Applying custom NPU placement:\n";
    
    // First, validate and apply the custom placement
    std::set<int> used_npu_ids;
    for (const auto& [coord, npu_id] : npu_placement) {
        int x = coord.first;
        int y = coord.second;
        
        // Check coordinate is in bounds
        if (x < 0 || x >= width || y < 0 || y >= height) {
            std::cerr << "[SPARSE-MESH2D-ERROR] NPU placement (" << x << "," << y << ") -> " 
                      << npu_id << " is out of bounds!\n";
            continue;
        }
        
        // Check coordinate is not excluded
        if (excluded.find({x, y}) != excluded.end()) {
            std::cerr << "[SPARSE-MESH2D-ERROR] NPU placement (" << x << "," << y << ") -> " 
                      << npu_id << " is at an excluded position!\n";
            continue;
        }
        
        // Check NPU ID is in valid range
        if (npu_id < 0 || npu_id >= valid_npu_count) {
            std::cerr << "[SPARSE-MESH2D-ERROR] NPU ID " << npu_id << " is out of range [0, " 
                      << valid_npu_count << ")!\n";
            continue;
        }
        
        // Check NPU ID is not already used
        if (used_npu_ids.count(npu_id)) {
            std::cerr << "[SPARSE-MESH2D-ERROR] NPU ID " << npu_id << " is assigned multiple times!\n";
            continue;
        }
        
        // Apply the mapping
        int grid_idx = coords_to_grid_index(x, y);
        grid_to_npu[grid_idx] = npu_id;
        npu_to_grid[npu_id] = {x, y};
        used_npu_ids.insert(npu_id);
        
        std::cerr << "  ├─ Position (" << x << "," << y << ") → NPU " << npu_id << "\n";
    }
    
    // Validate all NPU IDs are assigned
    if (used_npu_ids.size() != valid_npu_count) {
        std::cerr << "[SPARSE-MESH2D-ERROR] Expected " << valid_npu_count << " NPU placements, got " 
                  << used_npu_ids.size() << "!\n";
        // Fall back to auto-assignment for missing positions
        int next_auto_id = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (excluded.find({x, y}) == excluded.end()) {
                    int grid_idx = coords_to_grid_index(x, y);
                    if (grid_to_npu[grid_idx] < 0) {
                        // Find next unused ID
                        while (used_npu_ids.count(next_auto_id)) next_auto_id++;
                        grid_to_npu[grid_idx] = next_auto_id;
                        npu_to_grid[next_auto_id] = {x, y};
                        used_npu_ids.insert(next_auto_id);
                        std::cerr << "  ├─ [AUTO] Position (" << x << "," << y << ") → NPU " << next_auto_id << "\n";
                        next_auto_id++;
                    }
                }
            }
        }
    }

    std::cerr << "[SPARSE-MESH2D-INIT] Valid NPUs: " << valid_npu_count << "\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Bandwidth per link: " << bandwidth << " GB/s\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Latency per link: " << latency << " ns\n";

    // Fix topology metadata
    dims_count = 2;
    npus_count_per_dim.clear();
    npus_count_per_dim.push_back(width);
    npus_count_per_dim.push_back(height);
    bandwidth_per_dim.clear();
    bandwidth_per_dim.push_back(bandwidth);
    bandwidth_per_dim.push_back(bandwidth);

    basic_topology_type = TopologyBuildingBlock::Mesh2D;

    // Print grid layout with custom NPU IDs
    std::cerr << "\n[SPARSE-MESH2D-INIT] Grid layout (custom placement):\n\n";
    
    for (int y = 0; y < height; ++y) {
        std::cerr << "                  ";
        for (int x = 0; x < width; ++x) {
            int npu_id = get_npu_at(x, y);
            if (npu_id >= 0) {
                std::cerr << std::setw(3) << npu_id;
            } else {
                std::cerr << "  x";
            }
            
            if (x < width - 1) {
                int next_npu = get_npu_at(x + 1, y);
                if (npu_id >= 0 && next_npu >= 0) {
                    std::cerr << " --- ";
                } else {
                    std::cerr << "     ";
                }
            }
        }
        std::cerr << "\n";
        
        if (y < height - 1) {
            std::cerr << "                  ";
            for (int x = 0; x < width; ++x) {
                int npu_id = get_npu_at(x, y);
                int below_npu = get_npu_at(x, y + 1);
                if (npu_id >= 0 && below_npu >= 0) {
                    std::cerr << "  |  ";
                } else {
                    std::cerr << "     ";
                }
                if (x < width - 1) std::cerr << "     ";
            }
            std::cerr << "\n";
        }
    }
    std::cerr << "\n";

    // Create mesh links between adjacent valid nodes (same as before)
    int link_count = 0;
    std::cerr << "[SPARSE-MESH2D-INIT] Creating mesh links...\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int current_npu = get_npu_at(x, y);
            if (current_npu < 0) continue;

            // Connect to right neighbor
            if (x + 1 < width) {
                int right_npu = get_npu_at(x + 1, y);
                if (right_npu >= 0) {
                    connect(current_npu, right_npu, bandwidth, latency, true);
                    link_count += 2;
                    std::cerr << "[SPARSE-MESH2D-LINK] NPU " << current_npu << " (at " << x << "," << y << ") "
                              << "↔ NPU " << right_npu << " (at " << (x+1) << "," << y << ")\n";
                }
            }

            // Connect to bottom neighbor
            if (y + 1 < height) {
                int bottom_npu = get_npu_at(x, y + 1);
                if (bottom_npu >= 0) {
                    connect(current_npu, bottom_npu, bandwidth, latency, true);
                    link_count += 2;
                    std::cerr << "[SPARSE-MESH2D-LINK] NPU " << current_npu << " (at " << x << "," << y << ") "
                              << "↔ NPU " << bottom_npu << " (at " << x << "," << (y+1) << ")\n";
                }
            }
        }
    }
    
    std::cerr << "\n[SPARSE-MESH2D-INIT] Total bidirectional links created: " << (link_count / 2) << "\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Total directed links: " << link_count << "\n";
    std::cerr << "[SPARSE-MESH2D-INIT] Sparse Mesh2D topology (custom placement) COMPLETE ✓\n";
    std::cerr << "\n";

    // Print ring order - shows the CUSTOM order
    std::cerr << "[SPARSE-MESH2D-INIT] Ring order for collectives: ";
    for (int i = 0; i < valid_npu_count; ++i) {
        std::cerr << i;
        if (i < valid_npu_count - 1) std::cerr << " → ";
    }
    std::cerr << " → 0 (wrap)\n";
    
    // Also show physical path of ring
    std::cerr << "[SPARSE-MESH2D-INIT] Ring physical locations: ";
    for (int i = 0; i < valid_npu_count; ++i) {
        auto [x, y] = npu_to_grid[i];
        std::cerr << i << "@(" << x << "," << y << ")";
        if (i < valid_npu_count - 1) std::cerr << " → ";
    }
    std::cerr << " → 0 (wrap)\n\n";
}

bool SparseMesh2D::is_valid_position(int x, int y) const noexcept {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return false;
    }
    return excluded.find({x, y}) == excluded.end();
}

int SparseMesh2D::get_npu_at(int x, int y) const noexcept {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return -1;
    }
    return grid_to_npu[coords_to_grid_index(x, y)];
}

std::pair<int, int> SparseMesh2D::get_coords(DeviceId npu_id) const noexcept {
    assert(npu_id >= 0 && npu_id < valid_npu_count);
    return npu_to_grid[npu_id];
}

std::vector<std::pair<int, int>> SparseMesh2D::get_valid_neighbors(int x, int y) const noexcept {
    std::vector<std::pair<int, int>> neighbors;
    
    // Check all 4 directions
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};
    
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (is_valid_position(nx, ny)) {
            neighbors.push_back({nx, ny});
        }
    }
    
    return neighbors;
}

/**
 * Route using BFS to find shortest path (handles holes).
 */
Route SparseMesh2D::bfs_route(DeviceId src, DeviceId dest) const noexcept {
    auto [src_x, src_y] = get_coords(src);
    auto [dest_x, dest_y] = get_coords(dest);
    
    // BFS to find shortest path
    std::queue<std::pair<int, int>> q;
    std::map<std::pair<int, int>, std::pair<int, int>> parent;  // For path reconstruction
    std::set<std::pair<int, int>> visited;
    
    q.push({src_x, src_y});
    visited.insert({src_x, src_y});
    parent[{src_x, src_y}] = {-1, -1};  // Mark source
    
    while (!q.empty()) {
        auto [cx, cy] = q.front();
        q.pop();
        
        if (cx == dest_x && cy == dest_y) {
            // Found destination - reconstruct path
            Route route;
            std::vector<std::pair<int, int>> path;
            
            auto current = std::make_pair(dest_x, dest_y);
            while (current.first != -1) {
                path.push_back(current);
                current = parent[current];
            }
            
            // Reverse to get src → dest order
            std::reverse(path.begin(), path.end());
            
            for (const auto& [px, py] : path) {
                int npu_id = get_npu_at(px, py);
                route.push_back(devices[npu_id]);
            }
            
            return route;
        }
        
        // Explore neighbors
        for (const auto& [nx, ny] : get_valid_neighbors(cx, cy)) {
            if (visited.find({nx, ny}) == visited.end()) {
                visited.insert({nx, ny});
                parent[{nx, ny}] = {cx, cy};
                q.push({nx, ny});
            }
        }
    }
    
    // Should not reach here if mesh is connected
    std::cerr << "[SPARSE-MESH2D-ROUTE] ERROR: No path found from " << src << " to " << dest << "!\n";
    Route route;
    route.push_back(devices[src]);
    return route;
}

/**
 * Route function: Compute path between two NPUs
 *
 * Uses BFS for sparse meshes since XY routing may not work when there are holes.
 * BFS guarantees shortest path in terms of hop count.
 */
Route SparseMesh2D::route(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < valid_npu_count);
    assert(0 <= dest && dest < valid_npu_count);

    // Handle trivial case
    if (src == dest) {
        Route route;
        route.push_back(devices[src]);
        return route;
    }

    auto [src_x, src_y] = get_coords(src);
    auto [dest_x, dest_y] = get_coords(dest);

    std::cerr << "\n┌─────────────────────────────────────────────────────────────┐\n";
    std::cerr << "│          SPARSE MESH2D ROUTING - TRACE LOG                   │\n";
    std::cerr << "└─────────────────────────────────────────────────────────────┘\n";
    std::cerr << "[SPARSE-MESH2D-ROUTE] Source NPU: " << src << " at (" << src_x << ", " << src_y << ")\n";
    std::cerr << "[SPARSE-MESH2D-ROUTE] Destination NPU: " << dest << " at (" << dest_x << ", " << dest_y << ")\n";
    std::cerr << "[SPARSE-MESH2D-ROUTE] Using BFS for shortest path (handles holes)\n";

    Route route = bfs_route(src, dest);

    // Log the path
    std::cerr << "[SPARSE-MESH2D-ROUTE] Path: ";
    size_t idx = 0;
    for (const auto& device : route) {
        int npu_id = device->get_id();
        auto [x, y] = get_coords(npu_id);
        std::cerr << npu_id << "(" << x << "," << y << ")";
        if (idx < route.size() - 1) std::cerr << " → ";
        idx++;
    }
    std::cerr << "\n";
    std::cerr << "[SPARSE-MESH2D-ROUTE] Total hops: " << (route.size() - 1) << "\n";
    std::cerr << "[SPARSE-MESH2D-ROUTE] Latency: " << (latency * (route.size() - 1)) << " ns\n\n";

    return route;
}
