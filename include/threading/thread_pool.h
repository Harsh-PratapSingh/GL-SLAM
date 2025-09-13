#pragma once
#include "core/lightglue.h"
#include "core/superpoint.h"
#include "core/keypt2subpx.h"
#include "core/slam_core.h"
#include "core/slam_types.h"

namespace thread_pool {

    void tracking_thread();
    void map_optimizing_thread();

}