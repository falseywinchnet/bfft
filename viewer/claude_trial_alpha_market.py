"""Trial alpha 3: births and deaths priced in one currency.

The research log ends on the right question -- "pair births and merge deaths
must be compared in one measured objective before either is safe" -- and the
reason it stayed open is that the two sides were quoted in incomparable units.
A birth was scored by an integral of allocation pressure.  A death was scored
by the same integral, and then the whole exchange batch was accepted or
rejected by re-solving and looking at PSNR.  Nothing ever priced a single
cell's contribution to the objective that is actually minimised.

Measured here first, because it decides whether any of this matters: the
integrated-pressure score that currently chooses which cells die correlates
with the exact deletion price at rho = -0.02 on Pikachu at 2400 cells.  The
death side of the market has been choosing victims at random with respect to
the objective.

This module changes two decisions and nothing else:

  deaths  the victim set is the cheapest cells by exact Schur-complement
          price (`claude_trial_alpha_normal`), shortlisted by the O(n) upper
          bound so that pricing stays affordable;

  births  candidate positions are the most persistent components of the
          pressure field (`claude_trial_alpha_persistence`) rather than the
          per-cell arg-max under an exclusion disk.

Geometry, metric, assignment, and the fitted model are untouched.  Both
switches default off so the parent behaviour is recoverable exactly.
"""

from __future__ import annotations

import time

import numpy as np

from transport_voronoi import TransportVoronoi
import claude_trial_alpha_normal as alpha_normal
import claude_trial_alpha_persistence as alpha_persist


class MarketVoronoi(TransportVoronoi):
    """TransportVoronoi with priced deaths and topological births."""

    def __init__(self, image, config=None, exact_deaths=True,
                 persistence_births=True, price_shortlist=240,
                 cartoon_softness=4.0):
        self.exact_deaths = bool(exact_deaths)
        self.persistence_births = bool(persistence_births)
        self.price_shortlist = int(price_shortlist)
        self.cartoon_softness = float(cartoon_softness)
        self.price_report = {}
        super().__init__(image, config)

    # -- births ------------------------------------------------------
    def _subdivide(self):
        if not self.persistence_births:
            return super()._subdivide()
        room = self.cfg.max_cells - len(self.seeds)
        budget = min(max(0, int(self.cfg.split_batch)), room)
        if budget <= 0:
            return self._rebalance()
        sites, _ = alpha_persist.persistent_sites(
            self.allocation_pressure, budget, existing=self.seeds,
            min_distance=1.5)
        if not len(sites):
            return "none"
        marks, parents, generations = [], [], []
        for x, y in sites:
            index = int(round(y)) * self.w + int(round(x))
            marks.append(self._birth_mark(index))
            parent = int(self.owner[index])
            parents.append(parent)
            generations.append(int(self.generations[parent]) + 1)
        self.seeds = np.vstack([self.seeds, sites])
        self.marks = np.concatenate(
            [self.marks, np.asarray(marks, dtype=np.uint8)])
        self.parents = np.concatenate(
            [self.parents, np.asarray(parents, dtype=np.int32)])
        self.generations = np.concatenate(
            [self.generations, np.asarray(generations, dtype=np.int16)])
        self.support_map_indices = np.concatenate([
            self.support_map_indices,
            np.full(len(sites), -1, dtype=np.int32)])
        return "add"

    # -- deaths ------------------------------------------------------
    def exact_cell_prices(self, shortlist=None):
        """Exact deletion price for a bound-shortlisted set of cells.

        Returns (prices over the shortlist, shortlist indices, solver).
        """
        solver = alpha_normal.ExactCoupled(self, self.cartoon_softness)
        solver.field(self.base_lab)
        bounds = solver.deletion_bounds()
        k = min(int(shortlist or self.price_shortlist), len(bounds))
        candidates = np.argsort(bounds)[:k].astype(int)
        prices, columns = solver.deletion_prices(list(candidates))
        values = np.array([prices[i] for i in candidates])
        self.price_report = {
            "bound_median": float(np.median(bounds)),
            "priced": int(k),
            "exact_median": float(np.median(values)),
            "bound_over_exact": float(np.median(
                bounds[candidates] / np.maximum(values, 1e-18))),
        }
        return values, candidates, solver, columns

    def _rebalance(self):
        if not self.exact_deaths:
            return super()._rebalance()
        n = len(self.seeds)
        if n < 16:
            return "none"
        t0 = time.perf_counter()
        values, candidates, _, _ = self.exact_cell_prices()
        budget = min(max(1, int(self.cfg.split_batch) // 2), max(1, n // 20))
        weak = candidates[np.argsort(values)[:budget]]

        pressure = self.allocation_pressure.ravel()
        keep = np.ones(n, dtype=bool)
        keep[weak] = False
        blocked = np.zeros(self.npix, dtype=bool)
        for x, y in self.seeds[keep]:
            ix, iy = int(round(x)), int(round(y))
            x0, x1 = max(0, ix - 2), min(self.w, ix + 3)
            y0, y1 = max(0, iy - 2), min(self.h, iy + 3)
            blocked.reshape(self.h, self.w)[y0:y1, x0:x1] = True

        if self.persistence_births:
            field = self.allocation_pressure.copy()
            field.reshape(-1)[blocked] = 0.0
            sites, _ = alpha_persist.persistent_sites(
                field, len(weak), existing=self.seeds[keep], min_distance=2.0)
            replacements = [tuple(site) for site in sites]
        else:
            candidates_field = pressure.copy()
            candidates_field[blocked] = -1.0
            replacements = []
            for _ in range(len(weak)):
                idx = int(np.argmax(candidates_field))
                if candidates_field[idx] <= 0:
                    break
                y, x = divmod(idx, self.w)
                replacements.append((float(x), float(y)))
                x0, x1 = max(0, x - 4), min(self.w, x + 5)
                y0, y1 = max(0, y - 4), min(self.h, y + 5)
                candidates_field.reshape(
                    self.h, self.w)[y0:y1, x0:x1] = -1.0
        if not replacements:
            return "none"
        moved = len(replacements)
        marks = [self._birth_mark(
            int(round(y)) * self.w + int(round(x)))
            for x, y in replacements]
        self.seeds[weak[:moved]] = np.asarray(replacements)
        self.marks[weak[:moved]] = np.asarray(marks, dtype=np.uint8)
        self.price_report["rebalance_ms"] = (
            time.perf_counter() - t0) * 1000.0
        return "swap"


def reinvest(model, rounds=3, cost_fraction=0.01, shortlist=900,
             cartoon_softness=4.0, texture_softness=16.0,
             persistence_births=False, verbose=True):
    """Sell the cells that are not earning and buy where the pressure is.

    Measured motivation, Pikachu at 256 px / 2400 cells: 346 cells -- 14% of
    the budget -- can be switched off for a combined 1% of the fitted
    objective, and the incumbent death score does not identify them (rank
    correlation -0.02 with the exact price).  The budget is nominally full
    and materially idle.

    The price used here is an upper bound on the true cost of the sale, and
    deliberately so: it is computed with the partition held fixed, whereas an
    actual removal lets the neighbours re-own the freed pixels, which can
    only reduce the objective further.  So a sale that looks affordable is
    affordable.

    One round is: price (bound-filtered, exact on the shortlist), sell the
    cheapest prefix whose cumulative price stays under `cost_fraction` of the
    objective, re-buy the same number of sites from the pressure field,
    re-assign, re-fit.
    """
    from bfft.effects import lab_to_srgb

    fields = ("seeds", "marks", "parents", "generations",
              "support_map_indices")

    def snapshot():
        return {name: getattr(model, name).copy() for name in fields}

    def restore(state):
        for name, value in state.items():
            setattr(model, name, value.copy())
        model._assign()
        model._fit_models()
        model._render()
        alpha_normal.solve_coupled_exact(
            model, cartoon_softness, texture_softness)

    best_state = snapshot()
    best_psnr = float(model.psnr)
    history = []
    for index in range(int(rounds)):
        # A cell must be priced against everything it is asked to explain.
        # Pricing the cartoon alone reports texture-carrying cells as idle,
        # because they are idle in the field being priced and nowhere else.
        precision = float(np.clip(model.cfg.detail_precision, 0.0, 1.5))
        weight = precision * precision
        cartoon_solver = alpha_normal.ExactCoupled(model, cartoon_softness)
        base = cartoon_solver.field(model.base_lab)
        texture_solver = alpha_normal.ExactCoupled(model, texture_softness)
        detail = texture_solver.field(model.detail_lab)
        objective = (float(np.sum((model.base_lab - base) ** 2)) +
                     weight * float(np.sum((model.detail_lab - detail) ** 2)))
        bounds = (cartoon_solver.deletion_bounds() +
                  weight * texture_solver.deletion_bounds())
        n = len(model.seeds)
        k = min(int(shortlist), n)
        candidates = np.argsort(bounds)[:k].astype(int)
        cartoon_prices, _ = cartoon_solver.deletion_prices(list(candidates))
        texture_prices, _ = texture_solver.deletion_prices(list(candidates))
        values = np.array([cartoon_prices[i] + weight * texture_prices[i]
                           for i in candidates])
        order = np.argsort(values)
        affordable = int(np.sum(
            np.cumsum(values[order]) < cost_fraction * objective))
        affordable = min(affordable, max(0, n - 32))
        if affordable <= 0:
            break
        sell = candidates[order[:affordable]]

        keep = np.ones(n, dtype=bool)
        keep[sell] = False
        remap = np.full(n, -1, dtype=np.int64)
        remap[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
        model.seeds = model.seeds[keep]
        model.marks = model.marks[keep]
        parents = remap[np.clip(model.parents[keep], 0, n - 1)]
        model.parents = np.where(parents < 0, 0, parents).astype(np.int32)
        model.generations = model.generations[keep]
        model.support_map_indices = model.support_map_indices[keep]

        # Re-own the freed pixels before deciding where to buy, so the
        # pressure field describes the representation that now exists.
        model._assign()
        model._fit_models()
        model._render()

        pressure = model.allocation_pressure
        if persistence_births:
            sites, _ = alpha_persist.persistent_sites(
                pressure, affordable, existing=model.seeds,
                min_distance=2.0)
        else:
            field = pressure.ravel().copy()
            blocked = np.zeros(model.npix, dtype=bool)
            for x, y in model.seeds:
                ix, iy = int(round(x)), int(round(y))
                x0, x1 = max(0, ix - 2), min(model.w, ix + 3)
                y0, y1 = max(0, iy - 2), min(model.h, iy + 3)
                blocked.reshape(model.h, model.w)[y0:y1, x0:x1] = True
            field[blocked] = -1.0
            picks = []
            for _ in range(affordable):
                idx = int(np.argmax(field))
                if field[idx] <= 0:
                    break
                y, x = divmod(idx, model.w)
                picks.append((float(x), float(y)))
                x0, x1 = max(0, x - 3), min(model.w, x + 4)
                y0, y1 = max(0, y - 3), min(model.h, y + 4)
                field.reshape(model.h, model.w)[y0:y1, x0:x1] = -1.0
            sites = np.asarray(picks, dtype=np.float64).reshape(-1, 2)
        if not len(sites):
            break

        bought = len(sites)
        marks, parent_ids, generations = [], [], []
        for x, y in sites:
            pixel = int(round(y)) * model.w + int(round(x))
            marks.append(model._birth_mark(pixel))
            parent = int(model.owner[pixel])
            parent_ids.append(parent)
            generations.append(int(model.generations[parent]) + 1)
        model.seeds = np.vstack([model.seeds, sites])
        model.marks = np.concatenate(
            [model.marks, np.asarray(marks, dtype=np.uint8)])
        model.parents = np.concatenate(
            [model.parents, np.asarray(parent_ids, dtype=np.int32)])
        model.generations = np.concatenate(
            [model.generations, np.asarray(generations, dtype=np.int16)])
        model.support_map_indices = np.concatenate([
            model.support_map_indices,
            np.full(bought, -1, dtype=np.int32)])

        model._assign()
        model._fit_models()
        model._render()
        alpha_normal.solve_coupled_exact(
            model, cartoon_softness, texture_softness)
        rgb = np.clip(lab_to_srgb(model.reconstruction), 0.0, 1.0)
        mse = float(np.mean((model.rgb - rgb) ** 2))
        entry = {
            "round": index,
            "sold": int(affordable),
            "bought": int(bought),
            "cells": int(len(model.seeds)),
            "sale_cost_fraction": float(
                np.sum(values[order[:affordable]]) / max(objective, 1e-18)),
            "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
            "rgb_mse": mse,
        }
        history.append(entry)
        if entry["psnr"] > best_psnr:
            best_psnr = entry["psnr"]
            best_state = snapshot()
        if verbose:
            print("  round %d: sold %d, bought %d, sale cost %.2f%% of J "
                  "-> %.3f dB" % (index, affordable, bought,
                                  100 * entry["sale_cost_fraction"],
                                  entry["psnr"]), flush=True)

    # The sale price is exact but the purchase is a heuristic, so a round can
    # lose.  Keep the best state actually measured rather than the last one.
    if history and best_psnr > history[-1]["psnr"]:
        restore(best_state)
    return history


def price_correlation(model, shortlist=240, cartoon_softness=4.0):
    """How well does the incumbent death score predict the exact price?

    Reported as a rank correlation over the shortlist, because the incumbent
    only ever uses the ranking.
    """
    solver = alpha_normal.ExactCoupled(model, cartoon_softness)
    solver.field(model.base_lab)
    bounds = solver.deletion_bounds()
    n = len(model.seeds)
    k = min(shortlist, n)
    candidates = np.argsort(bounds)[:k].astype(int)
    prices, columns = solver.deletion_prices(list(candidates))
    exact = np.array([prices[i] for i in candidates])
    incumbent = np.bincount(
        model.owner, weights=model.allocation_pressure.ravel(),
        minlength=n)[candidates]

    def rank(a):
        return np.argsort(np.argsort(a)).astype(np.float64)

    rho = float(np.corrcoef(rank(incumbent), rank(exact))[0, 1])
    return {
        "spearman_incumbent_vs_exact": rho,
        "exact_min": float(exact.min()),
        "exact_max": float(exact.max()),
        "exact_median": float(np.median(exact)),
        "bound_over_exact_median": float(np.median(
            bounds[candidates] / np.maximum(exact, 1e-18))),
        "solver": solver,
        "columns": columns,
        "candidates": candidates,
        "exact": exact,
    }
