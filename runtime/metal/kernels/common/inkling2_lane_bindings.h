#pragma once

// Eight-lane buffer bindings for the ported Inkling-2 kernels. The downstream
// kernel ABI binds four lanes (SPLASH_MAXIMUM_BATCH_WIDTH == 4, see
// kernel/common/lane_bindings.h) and that macro stays untouched for the
// existing kernels. The ported kernels bind eight state/page-table buffers so
// that the same source compiles as upstream; indices past the fork's lane
// count are never selected by a dispatched lane. Keep this separate so the
// fork's four-lane kernels do not see a second definition of their macro.
#define SPLASH_LANE_BINDING_8(lane, lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7)                  \
  ((lane) == 0   ? (lane0)                                                                                   \
   : (lane) == 1 ? (lane1)                                                                                   \
   : (lane) == 2 ? (lane2)                                                                                   \
   : (lane) == 3 ? (lane3)                                                                                   \
   : (lane) == 4 ? (lane4)                                                                                   \
   : (lane) == 5 ? (lane5)                                                                                   \
   : (lane) == 6 ? (lane6)                                                                                   \
                 : (lane7))
