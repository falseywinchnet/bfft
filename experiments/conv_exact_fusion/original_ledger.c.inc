            double best_cost = INFINITY;
            int best_boundary = begin;
            for (int candidate = begin; candidate <= end; ++candidate) {
                double cost = 0.0;
                for (int slot = local_begin; slot <= local_end; ++slot) {
                    const int8_t expected = slot < candidate ? coarse[knot-1] : coarse[knot];
                    const int cell = slot / 5, k = slot % 5;
                    const float value = current[(cell*5+k)*lanes+lane];
                    if (expected * value < 0.0f) cost += (double)value * value;
                }
                if (cost < best_cost) { best_cost = cost; best_boundary = candidate; }
            }
