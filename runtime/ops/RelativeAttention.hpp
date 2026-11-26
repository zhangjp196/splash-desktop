#pragma once

// Upstream's header name for the Inkling-2 relative attention operator. The
// ported implementation lives in the inkling2_ prefixed header so the fork's
// four-lane attention ops are unchanged; the family sources use this name.
#include "ops/inkling2_RelativeAttention.hpp"
