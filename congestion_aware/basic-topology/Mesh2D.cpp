/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Mesh2D.h"
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <ostream>
#include <iomanip> // Required for std::setw

using namespace NetworkAnalyticalCongestionAware;

/**
 * Constructor: Initialize 2D Mesh Topology
 *
 * This constructor creates a 2D mesh grid with width columns and height rows.
 * Each internal node connects to 4 neighbors (up, down, left, right).
 * Edge nodes connect to fewer neighbors (no wrap-around).
 *
 * Example: Mesh2D(4, 3) creates:
 *     0 --- 1 --- 2 --- 3
 *     |     |     |     |
 *     4 --- 5 --- 6 --- 7
 *     |     |     |     |
 *     8 --- 9 --- 10--- 11
 *
 * Total NPUs = width × height = 4 × 3 = 12
 * Total devices = 12 (no extra switch nodes, unlike Switch topology)
 * Total links = 2 × (width × (height-1) + height × (width-1))
 *             = 2 × (4 × 2 + 3 × 3) = 2 × 17 = 34 directed links (bidirectional)
 */
Mesh2D::Mesh2D(const int width, const int height, const Bandwidth bandwidth, const Latency latency) noexcept
    : width(width), height(height), BasicTopology(width * height, width * height, bandwidth, latency) {
    
    // Validate input parameters
    assert(width > 0);
    assert(height > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // Fix topology metadata to reflect 2D mesh (so frontends see both dimensions)
    dims_count = 2;
    npus_count_per_dim.clear();
    npus_count_per_dim.push_back(width);   // X dimension
    npus_count_per_dim.push_back(height);  // Y dimension
    bandwidth_per_dim.clear();
    bandwidth_per_dim.push_back(bandwidth);  // bandwidth per dim (uniform)
    bandwidth_per_dim.push_back(bandwidth);

    // =====================================================
    // MESH2D TOPOLOGY CONSTRUCTION - DETAILED LOGGING
    // =====================================================
    std::cerr << "\n";
    std::cerr << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║                  MESH2D TOPOLOGY INITIALIZATION                            ║\n";
    std::cerr << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    
    std::cerr << "\n[MESH2D-INIT] Dimensions: " << width << " (width) × " << height << " (height)\n";
    std::cerr << "[MESH2D-INIT] Total NPUs: " << (width * height) << "\n";
    std::cerr << "[MESH2D-INIT] Bandwidth per link: " << bandwidth << " GB/s\n";
    std::cerr << "[MESH2D-INIT] Latency per link: " << latency << " ns\n";

    // Set the topology type identifier
    basic_topology_type = TopologyBuildingBlock::Mesh2D;

    /**
     * Connect mesh nodes in 4-neighbor topology
     * 
     * Strategy:
     * - Iterate through each position in the grid
     * - Connect each node to its right neighbor (if exists)
     * - Connect each node to its bottom neighbor (if exists)
     * - Use bidirectional connections so reverse links are automatic
     * 
     * This avoids duplicate connections and creates the proper mesh structure
     */
    
    int link_count = 0;
    std::cerr << "\n[MESH2D-INIT] Creating mesh links...\n";
    std::cerr << "[MESH2D-INIT] Grid layout:\n\n";
    
    for (int y = 0; y < height; ++y) {
        std::cerr << "                  ";
        for (int x = 0; x < width; ++x) {
            std::cerr << std::setw(3) << (y * width + x);
            if (x < width - 1) std::cerr << " --- ";
        }
        std::cerr << "\n";
        
        if (y < height - 1) {
            std::cerr << "                  ";
            for (int x = 0; x < width; ++x) {
                std::cerr << "  |  ";
                if (x < width - 1) std::cerr << "     ";
            }
            std::cerr << "\n";
        }
    }
    std::cerr << "\n";

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Current node ID in linear indexing: y * width + x
            DeviceId current = coords_to_npu_id(x, y);

            // Connect to right neighbor (x + 1, y) if it exists
            if (x + 1 < width) {
                DeviceId right = coords_to_npu_id(x + 1, y);
                // Bidirectional connection: both current↔right
                connect(current, right, bandwidth, latency, true);
                link_count += 2;  // bidirectional = 2 directed links
                std::cerr << "[MESH2D-LINK] NPU " << current << " (at " << x << "," << y << ") "
                          << "↔ NPU " << right << " (at " << (x+1) << "," << y << ")\n";
            }

            // Connect to bottom neighbor (x, y + 1) if it exists
            if (y + 1 < height) {
                DeviceId bottom = coords_to_npu_id(x, y + 1);
                // Bidirectional connection: both current↔bottom
                connect(current, bottom, bandwidth, latency, true);
                link_count += 2;  // bidirectional = 2 directed links
                std::cerr << "[MESH2D-LINK] NPU " << current << " (at " << x << "," << y << ") "
                          << "↔ NPU " << bottom << " (at " << x << "," << (y+1) << ")\n";
            }

            // Top and left neighbors are implicitly connected through previous iterations
            // - When we processed (x-1, y), it connected to current
            // - When we processed (x, y-1), it connected to current
        }
    }
    
    std::cerr << "\n[MESH2D-INIT] Total bidirectional links created: " << (link_count / 2) << "\n";
    std::cerr << "[MESH2D-INIT] Total directed links: " << link_count << "\n";
    std::cerr << "[MESH2D-INIT] Mesh2D topology construction COMPLETE ✓\n";
    std::cerr << "\n";
}

/**
 * Simplified Constructor: Initialize 2D Mesh Topology from NPU Count
 * 
 * This constructor automatically derives width and height from total npus_count.
 * For square meshes (e.g., 16 NPUs → 4×4), it calculates the square root.
 * For non-square meshes, it approximates the closest square dimensions.
 * 
 * @param npus_count total number of NPUs (should be a perfect square ideally)
 * @param bandwidth bandwidth per link
 * @param latency latency per link
 */
Mesh2D::Mesh2D(const int npus_count, const Bandwidth bandwidth, const Latency latency) noexcept
    : Mesh2D(static_cast<int>(std::sqrt(npus_count)), 
             static_cast<int>(std::sqrt(npus_count)),
             bandwidth, 
             latency) {
    // Validate that npus_count is a perfect square
    const int width_height = static_cast<int>(std::sqrt(npus_count));
    if (width_height * width_height != npus_count) {
        std::cerr << "\n⚠️  [MESH2D-CONSTRUCTOR] WARNING: npus_count " << npus_count 
                  << " is NOT a perfect square!\n";
        std::cerr << "[MESH2D-CONSTRUCTOR] Using approximation: " << width_height << "×" << width_height 
                  << " = " << (width_height * width_height) << " NPUs\n";
        std::cerr << "[MESH2D-CONSTRUCTOR] Lost NPUs: " << (npus_count - (width_height * width_height)) << "\n";
        std::cerr << "[MESH2D-CONSTRUCTOR] For perfect square, use: ";
        int next_square = (width_height + 1) * (width_height + 1);
        int prev_square = width_height * width_height;
        std::cerr << prev_square << " or " << next_square << "\n\n";
    }
}

/**
 * Route function: Compute path between two NPUs using XY routing
 *
 * XY Routing Algorithm:
 * 1. Decompose source and destination into (x, y) coordinates
 * 2. Move from src_x to dest_x along X dimension (left/right)
 * 3. Move from src_y to dest_y along Y dimension (up/down)
 * 4. Collect all intermediate nodes into a route
 *
 * Properties:
 * - Deterministic: Always produces same route for same (src, dest) pair
 * - Deadlock-free: No circular wait when combined with flow control
 * - Optimal path length: Manhattan distance = hop count
 * - Simple to implement and understand
 *
 * Example: 4×3 mesh, NPU 0 → NPU 11
 *   src: 0 → (0, 0)
 *   dest: 11 → (3, 2)
 *   Route:
 *     Step 0: At (0, 0) → NPU 0
 *     Step 1: Move X: (0, 0) → (1, 0) → NPU 1
 *     Step 2: Move X: (1, 0) → (2, 0) → NPU 2
 *     Step 3: Move X: (2, 0) → (3, 0) → NPU 3
 *     Step 4: Move Y: (3, 0) → (3, 1) → NPU 7
 *     Step 5: Move Y: (3, 1) → (3, 2) → NPU 11
 *   Total hops: 5 = |3-0| + |2-0| = Manhattan distance ✓
 */
Route Mesh2D::route(const DeviceId src, const DeviceId dest) const noexcept {
    // Validate source and destination are in valid range
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    // Handle trivial case: source == destination
    if (src == dest) {
        Route route;
        route.push_back(devices[src]);
        std::cerr << "[MESH2D-ROUTE] Same source/dest (NPU " << src << ") - No routing needed\n";
        return route;
    }

    // Decompose source and destination into 2D coordinates
    auto [src_x, src_y] = get_2d_coords(src);
    auto [dest_x, dest_y] = get_2d_coords(dest);

    // Initialize route with source node
    Route route;
    route.push_back(devices[src]);

    std::cerr << "\n┌─────────────────────────────────────────────────────────────┐\n";
    std::cerr << "│              MESH2D XY ROUTING - TRACE LOG                   │\n";
    std::cerr << "└─────────────────────────────────────────────────────────────┘\n";
    std::cerr << "[MESH2D-ROUTE] Source NPU: " << src << " at coordinates (" << src_x << ", " << src_y << ")\n";
    std::cerr << "[MESH2D-ROUTE] Destination NPU: " << dest << " at coordinates (" << dest_x << ", " << dest_y << ")\n";
    std::cerr << "[MESH2D-ROUTE] Mesh dimensions: " << width << " × " << height << "\n";
    std::cerr << "[MESH2D-ROUTE] Manhattan distance: |" << dest_x << " - " << src_x << "| + |" << dest_y 
              << " - " << src_y << "| = " << (std::abs(dest_x - src_x) + std::abs(dest_y - src_y)) << " hops\n\n";

    /**
     * PHASE 1: Move along X dimension (left or right)
     */
    int x = src_x;
    int y = src_y;

    std::cerr << "═══ PHASE 1: Move along X dimension ═══\n";
    std::cerr << "[MESH2D-ROUTE] Current position: (" << x << ", " << y << ")\n";
    std::cerr << "[MESH2D-ROUTE] Target X: " << dest_x << "\n";
    
    int phase1_hops = 0;
    while (x != dest_x) {
        // Determine direction: +1 for right, -1 for left
        int direction = (dest_x > x) ? 1 : -1;
        std::string dir_str = (direction > 0) ? "RIGHT" : "LEFT";
        
        x += direction;
        phase1_hops++;
        
        // Add the next node to route
        DeviceId next_npu = coords_to_npu_id(x, y);
        route.push_back(devices[next_npu]);
        
        std::cerr << "[MESH2D-ROUTE] Hop " << phase1_hops << ": Move " << dir_str 
                  << " → NPU " << next_npu << " at (" << x << ", " << y << ")\n";
    }
    
    std::cerr << "[MESH2D-ROUTE] Phase 1 complete: " << phase1_hops << " hops in X direction\n\n";

    /**
     * PHASE 2: Move along Y dimension (up or down)
     */
    std::cerr << "═══ PHASE 2: Move along Y dimension ═══\n";
    std::cerr << "[MESH2D-ROUTE] Current position: (" << x << ", " << y << ")\n";
    std::cerr << "[MESH2D-ROUTE] Target Y: " << dest_y << "\n";
    
    int phase2_hops = 0;
    while (y != dest_y) {
        // Determine direction: +1 for down, -1 for up
        int direction = (dest_y > y) ? 1 : -1;
        std::string dir_str = (direction > 0) ? "DOWN" : "UP";
        
        y += direction;
        phase2_hops++;
        
        // Add the next node to route
        DeviceId next_npu = coords_to_npu_id(x, y);
        route.push_back(devices[next_npu]);
        
        std::cerr << "[MESH2D-ROUTE] Hop " << (phase1_hops + phase2_hops) << ": Move " << dir_str 
                  << " → NPU " << next_npu << " at (" << x << ", " << y << ")\n";
    }
    
    std::cerr << "[MESH2D-ROUTE] Phase 2 complete: " << phase2_hops << " hops in Y direction\n\n";

    // Summary
    std::cerr << "╔═════════════════════════════════════════════════════════════╗\n";
    std::cerr << "║                 ROUTE SUMMARY                               ║\n";
    std::cerr << "╚═════════════════════════════════════════════════════════════╝\n";
    std::cerr << "[MESH2D-ROUTE] Total route length: " << route.size() << " nodes (including source and destination)\n";
    std::cerr << "[MESH2D-ROUTE] Total hops: " << (route.size() - 1) << "\n";
    std::cerr << "[MESH2D-ROUTE] Path breakdown: X=" << phase1_hops << " hops + Y=" << phase2_hops << " hops\n";
    std::cerr << "[MESH2D-ROUTE] Latency (link): " << latency << " ns per hop × " << (route.size() - 1) 
              << " hops = " << (latency * (route.size() - 1)) << " ns total\n";
    std::cerr << "\n";

    // At this point, we're at (dest_x, dest_y) which is the destination
    // Return the complete route
    return route;
}

/**
 * Helper function: Check if two NPUs are direct neighbors
 *
 * Two NPUs are neighbors if their Manhattan distance is exactly 1.
 * This means they share an edge (not diagonal).
 *
 * Neighbor relationships:
 * - (x, y) ↔ (x+1, y): right neighbor
 * - (x, y) ↔ (x-1, y): left neighbor
 * - (x, y) ↔ (x, y+1): bottom neighbor
 * - (x, y) ↔ (x, y-1): top neighbor
 *
 * NOT neighbors (diagonal):
 * - (x, y) ✗ (x+1, y+1): distance = 2
 *
 * @param src source NPU ID
 * @param dest destination NPU ID
 * @return true if Manhattan distance = 1, false otherwise
 */
bool Mesh2D::are_neighbors(const DeviceId src, const DeviceId dest) const noexcept {
    return manhattan_distance(src, dest) == 1;
}

/**
 * Helper function: Calculate Manhattan distance between two NPUs
 *
 * Manhattan distance = |dest_x - src_x| + |dest_y - src_y|
 *
 * Examples (4×3 mesh):
 * - Distance from 0 (0,0) to 1 (1,0): |1-0| + |0-0| = 1
 * - Distance from 0 (0,0) to 5 (1,1): |1-0| + |1-0| = 2
 * - Distance from 0 (0,0) to 11 (3,2): |3-0| + |2-0| = 5
 *
 * Properties:
 * - Symmetric: distance(A, B) = distance(B, A)
 * - Triangle inequality: distance(A, C) ≤ distance(A, B) + distance(B, C)
 * - Optimal hop count in mesh: hops = Manhattan distance
 *
 * @param src source NPU ID
 * @param dest destination NPU ID
 * @return Manhattan distance (non-negative integer)
 */
int Mesh2D::manhattan_distance(const DeviceId src, const DeviceId dest) const noexcept {
    auto [src_x, src_y] = get_2d_coords(src);
    auto [dest_x, dest_y] = get_2d_coords(dest);
    
    // Calculate absolute differences in each dimension
    int dx = std::abs(dest_x - src_x);
    int dy = std::abs(dest_y - src_y);
    
    // Return sum (Manhattan distance)
    return dx + dy;
}
