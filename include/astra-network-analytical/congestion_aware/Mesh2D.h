/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/BasicTopology.h"
#include <utility>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Implements a 2D Mesh topology.
 *
 * Mesh2D(4, 3) example with width=4, height=3:
 *
 *     0 --- 1 --- 2 --- 3
 *     |     |     |     |
 *     4 --- 5 --- 6 --- 7
 *     |     |     |     |
 *     8 --- 9 --- 10--- 11
 *
 * Each NPU connects to its neighbors in 4 directions (up, down, left, right).
 * Edge NPUs have fewer connections (no wrap-around, unlike torus).
 *
 * Connectivity:
 * - Internal nodes (5, 6, 9, 10): 4 neighbors each
 * - Edge nodes (1, 2, 4, 7, 8, 11): 3 neighbors each
 * - Corner nodes (0, 3, 8, 11): 2 neighbors each
 *
 * Routing: XY routing algorithm
 * - Move first in X direction, then Y direction
 * - For NPU 0 to NPU 11: 0→1→2→3→7→11 (move right, then down)
 * - Hops = Manhattan distance = |dest_x - src_x| + |dest_y - src_y|
 *
 * The number of devices equals number of NPUs (no extra switch nodes).
 */
class Mesh2D final : public BasicTopology {
  public:
    /**
     * Constructor for 2D Mesh topology.
     *
     * @param width number of nodes in X dimension (columns)
     * @param height number of nodes in Y dimension (rows)
     * @param bandwidth bandwidth per link (GB/s)
     * @param latency latency per link (nanoseconds)
     */
    Mesh2D(int width, int height, Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Constructor for 2D Mesh topology (simplified).
     * Automatically derives width and height from npus_count.
     * Assumes a square mesh when possible (e.g., 16 NPUs → 4×4).
     *
     * @param npus_count total number of NPUs (width × height must equal this)
     * @param bandwidth bandwidth per link (GB/s)
     * @param latency latency per link (nanoseconds)
     */
    Mesh2D(int npus_count, Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Compute route between two NPUs using XY routing algorithm.
     *
     * @param src source NPU ID
     * @param dest destination NPU ID
     * @return sequence of devices (nodes) to traverse from src to dest
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

  private:
    /// Width of mesh (number of columns)
    int width;

    /// Height of mesh (number of rows)
    int height;

    /**
     * Convert linear NPU ID to 2D coordinates (x, y).
     *
     * For width=4, height=3:
     * NPU 0 → (0, 0)
     * NPU 1 → (1, 0)
     * NPU 4 → (0, 1)
     * NPU 5 → (1, 1)
     *
     * @param npu_id linear NPU identifier
     * @return pair (x, y) where x ∈ [0, width-1], y ∈ [0, height-1]
     */
    [[nodiscard]] std::pair<int, int> get_2d_coords(DeviceId npu_id) const noexcept {
        return {npu_id % width, npu_id / width};
    }

    /**
     * Convert 2D coordinates to linear NPU ID.
     *
     * @param x column coordinate
     * @param y row coordinate
     * @return linear NPU ID = y * width + x
     */
    [[nodiscard]] DeviceId coords_to_npu_id(int x, int y) const noexcept {
        return y * width + x;
    }

    /**
     * Check if two NPUs are direct neighbors (1 hop distance).
     * Used for validation and debugging.
     *
     * @param src source NPU ID
     * @param dest destination NPU ID
     * @return true if they are adjacent (Manhattan distance = 1), false otherwise
     */
    [[nodiscard]] bool are_neighbors(DeviceId src, DeviceId dest) const noexcept;

    /**
     * Calculate Manhattan distance between two NPUs.
     * Used for hop count estimation.
     *
     * @param src source NPU ID
     * @param dest destination NPU ID
     * @return Manhattan distance = |dest_x - src_x| + |dest_y - src_y|
     */
    [[nodiscard]] int manhattan_distance(DeviceId src, DeviceId dest) const noexcept;
};

}  // namespace NetworkAnalyticalCongestionAware
