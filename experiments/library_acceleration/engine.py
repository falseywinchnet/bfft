"""Use the existing finite-pass engine with counted, sampled admission.

The existing discover() screen is a falsification heuristic, not a path
certificate. No ordinary future, reference image, or output score informs
admission. Conservation of requested iteration count includes the settling
pass. A rejected attempt returns its already-computed next ordinary state.
"""
from experiments.krylov_bregman.discovery import discover


def run(model, passes, accelerated=True, tolerance=.005, warmup=8,
        cooldown=16, depths=(2, 4), maximum_horizon=64):
    state = model.initial.copy()
    count = calls = actions = accepted = rejected = skipped = 0
    next_attempt = warmup
    events = []
    while count < passes:
        remaining = passes - count
        horizons = tuple(h for h in (64, 32, 16) if h <= maximum_horizon and h + 1 <= remaining)
        if accelerated and count >= next_attempt and horizons:
            try:
                result = discover(model, state, depths=depths, horizons=horizons,
                                  tolerance=tolerance)
            except ValueError:
                state = model.step(state); calls += 1; count += 1
                rejected += 1; next_attempt = count + cooldown
                continue
            calls += result.map_calls; actions += result.tangent_actions
            if result.accepted and model.valid(result.candidate):
                # One settling pass is included in the remaining budget.
                state = model.step(result.candidate); calls += 1
                count += result.horizon + 1
                skipped += result.horizon - 1; accepted += 1
                events.append(dict(at=count-result.horizon-1, horizon=result.horizon,
                                   depth=result.depth, score=result.sampled_score))
                next_attempt = count
            else:
                if result.accepted:
                    state = model.step(state); calls += 1
                else:
                    state = result.candidate
                count += 1; rejected += 1; next_attempt = count + cooldown
        else:
            state = model.step(state); calls += 1; count += 1
    return model.output(state), dict(passes=count, map_calls=calls, tangent_actions=actions,
                                    accepted=accepted, rejected=rejected, skipped=skipped, events=events)
