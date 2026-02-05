/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/BasicTopology.h"
#include <utility>
#include <vector>
#include <set>
#include <map>
#include <cstdlib>  // for std::abs

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Implements a Sparse 2D Mesh topology with support for excluded/missing nodes.
 *
 * This allows creating irregular mesh shapes where some grid positions are "holes".
 * Valid nodes are numbered contiguously (0, 1, 2, ...) regardless of their grid position.
 *
 * Example: SparseMesh2D(6, 4) with excluded positions creates:
 *
 *     0 --- 1 --- 2 --- 3 --- 4 --- 5
 *     x     x     |     |     |     |
 *     x     x     6 --- 7 --- 8 --- 9
 *     x     x     |     |     |     |
 *     x     x    10 ---11 ---12 ---13
 *     x     x     x     x     |     |
 *     x     x     x     x    14 ---15
 *
 * Here 'x' represents excluded positions. Valid NPUs are numbered 0-15 contiguously.
 *
 * Routing uses a modified XY algorithm that navigates around holes:
 * - Tries XY routing first (move X, then Y)
 * - If blocked by a hole, uses BFS to find shortest path
 *
 * The ring for collective communication visits all valid nodes in order: 0→1→2→...→15→0
 */
class SparseMesh2D final : public BasicTopology {
  public:
    /**
     * Constructor for Sparse 2D Mesh topology with automatic NPU numbering.
     *
     * @param width maximum number of columns in the grid
     * @param height maximum number of rows in the grid
     * @param excluded_coords set of (x, y) coordinates that should NOT have nodes
     * @param bandwidth bandwidth per link (GB/s)
     * @param latency latency per link (nanoseconds)
     */
    SparseMesh2D(int width, int height, 
                 const std::set<std::pair<int, int>>& excluded_coords,
                 Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Constructor for Sparse 2D Mesh topology with CUSTOM NPU placement.
     * 
     * This allows defining custom NPU ID assignments for optimized routing patterns
     * (e.g., snake patterns where ring neighbors are physically adjacent).
     *
     * @param width maximum number of columns in the grid
     * @param height maximum number of rows in the grid
     * @param excluded_coords set of (x, y) coordinates that should NOT have nodes
     * @param npu_placement map from (x, y) to NPU ID for custom assignment
     * @param bandwidth bandwidth per link (GB/s)
     * @param latency latency per link (nanoseconds)
     */
    SparseMesh2D(int width, int height, 
                 const std::set<std::pair<int, int>>& excluded_coords,
                 const std::map<std::pair<int, int>, int>& npu_placement,
                 Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Compute route between two NPUs.
     * Uses modified XY routing that navigates around holes.
     *
     * @param src source NPU ID (in contiguous numbering)
     * @param dest destination NPU ID (in contiguous numbering)
     * @return sequence of devices (nodes) to traverse from src to dest
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

    /**
     * Get the number of valid (non-excluded) NPUs.
     */
    [[nodiscard]] int get_valid_npu_count() const noexcept { return valid_npu_count; }

    /**
     * Get the grid width.
     */
    [[nodiscard]] int get_width() const noexcept { return width; }

    /**
     * Get the grid height.
     */
    [[nodiscard]] int get_height() const noexcept { return height; }

    /**
     * Check if a grid position is valid (not excluded).
     */
    [[nodiscard]] bool is_valid_position(int x, int y) const noexcept;

    /**
     * Get NPU ID for a grid position (-1 if excluded).
     */
    [[nodiscard]] int get_npu_at(int x, int y) const noexcept;

    /**
     * Get grid coordinates for an NPU ID.
     */
    [[nodiscard]] std::pair<int, int> get_coords(DeviceId npu_id) const noexcept;

  private:
    /// Maximum width of grid (number of columns)
    int width;

    /// Maximum height of grid (number of rows)
    int height;

    /// Number of valid (non-excluded) NPUs
    int valid_npu_count;

    /// Set of excluded coordinates
    std::set<std::pair<int, int>> excluded;

    /// Map from grid coordinates to NPU ID (-1 if excluded)
    /// Index: y * width + x
    std::vector<int> grid_to_npu;

    /// Map from NPU ID to grid coordinates
    std::vector<std::pair<int, int>> npu_to_grid;

    /**
     * Convert grid coordinates to linear index.
     */
    [[nodiscard]] int coords_to_grid_index(int x, int y) const noexcept {
        return y * width + x;
    }

    /**
     * Check if two grid positions are adjacent (Manhattan distance = 1).
     */
    [[nodiscard]] bool are_adjacent(int x1, int y1, int x2, int y2) const noexcept {
        return (std::abs(x1 - x2) + std::abs(y1 - y2)) == 1;
    }

    /**
     * Get valid neighbors of a grid position.
     * Returns list of (x, y) coordinates of valid adjacent nodes.
     */
    [[nodiscard]] std::vector<std::pair<int, int>> get_valid_neighbors(int x, int y) const noexcept;

    /**
     * Find shortest path using BFS when XY routing is blocked.
     */
    [[nodiscard]] Route bfs_route(DeviceId src, DeviceId dest) const noexcept;
};

}  // namespace NetworkAnalyticalCongestionAware
