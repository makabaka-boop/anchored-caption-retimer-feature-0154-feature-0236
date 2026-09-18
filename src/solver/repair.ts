// Repair: when pinned cues are jointly infeasible, compute the largest set of
// pins that can be kept so the solver becomes feasible again.
//
// In the transformed coordinate y = pinStart - P[cueIndex] (P = duration
// prefix) a kept set is feasible iff
//   1. every kept pin satisfies 0 <= y <= daySpan - P[n-1]   (individual box)
//   2. the kept y values are non-decreasing in cue order     (chain)
// Condition 1 is per-pin, so a pin violating it belongs to no feasible set at
// all and is necessarily released. Among the remaining pins condition 2 is a
// longest non-decreasing subsequence (LNDS) problem on the clean y sequence.
//
// Tie-break: with several maximum sets, the released list (forced ∪ chosen,
// cueIndex ascending) must be lexicographically smallest. For equal-size
// subsets of a fixed ordered universe, sorted-release lexicographic minimum
// is exactly sorted-kept lexicographic maximum (the first symmetric-difference
// element decides both comparisons in opposite directions), so we compute the
// lexicographically largest — by position — longest NDS:
//   L2[i] = length of the longest NDS starting at i, via a suffix-max Fenwick
//   over compressed y values; then for remaining lengths L..1 greedily take
//   the rightmost position after the previous pick whose y is not smaller.
//   Each level bucket is scanned once, so selection is O(k) after the
//   O(k log k) sort/compression/L2 phase: O(k log k) time and O(k) space in
//   k pins, with no repeated solve() calls.

import { DAY_MS, type Cue, type Pins } from './solve';

export interface RepairInput {
  cues: ReadonlyArray<Cue>;
  pins: Pins;
  /** Day span override (tests use small spans); defaults to the full day. */
  daySpan?: number;
}

export interface RepairAnalysis {
  /** Pins to keep: cueIndex -> fixed start (a subset of the input pins). */
  keep: Map<number, number>;
  /** Complete cueIndex-ascending release list (forced ∪ chosen). */
  unpin: number[];
  /** Ascending cueIndices no feasible kept set can contain (box violators). */
  forced: number[];
  /** keep.size, exposed for the preview line. */
  keptCount: number;
}

export type RepairResult =
  | { ok: true; analysis: RepairAnalysis }
  | { ok: false; reason: 'DAY_OVERFLOW' };

/**
 * Lexicographically largest (by position) longest non-decreasing subsequence
 * of y, returned as an ascending list of positions. O(m log m) time, O(m)
 * space.
 */
function longestKept(y: ReadonlyArray<number>): number[] {
  const m = y.length;
  if (m === 0) return [];

  // Coordinate compression.
  const values = Array.from(new Set(y)).sort((a, b) => a - b);
  const u = values.length;
  const comp = new Array<number>(m);
  for (let i = 0; i < m; i++) {
    let lo = 0;
    let hi = u - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (values[mid] < y[i]) lo = mid + 1;
      else hi = mid;
    }
    comp[i] = lo;
  }

  // L2[i] = 1 + max{ L2[j] : j > i, y[j] >= y[i] }, computed right to left.
  // The Fenwick tree answers prefix maxima on the reversed coordinate
  // rc = u-1-comp, where y[j] >= y[i] becomes rc[j] <= rc[i].
  const bit = new Array<number>(u + 1).fill(0);
  const L2 = new Array<number>(m);
  let L = 0;
  for (let i = m - 1; i >= 0; i--) {
    const rc = u - 1 - comp[i];
    let best = 0;
    for (let x = rc + 1; x > 0; x -= x & -x) if (bit[x] > best) best = bit[x];
    const v = best + 1;
    L2[i] = v;
    if (v > L) L = v;
    for (let x = rc + 1; x <= u; x += x & -x) if (v > bit[x]) bit[x] = v;
  }

  // Buckets of positions by level, ascending (built in one forward pass).
  const levels: number[][] = Array.from({ length: L + 1 }, () => []);
  for (let i = 0; i < m; i++) levels[L2[i]].push(i);

  // Greedy lex-max: for remaining lengths L..1 take the rightmost position
  // after the previous pick with y not smaller. A valid continuation always
  // exists: the previous pick's own length-(ell+1) chain continues through a
  // position with L2 = ell, and L2 > ell there would contradict maximality
  // of L. Buckets are disjoint and each is scanned once: O(m) total.
  const kept: number[] = [];
  let prevPos = -1;
  let prevY = -Infinity;
  for (let ell = L; ell >= 1; ell--) {
    const bucket = levels[ell];
    let picked = false;
    for (let t = bucket.length - 1; t >= 0; t--) {
      const j = bucket[t];
      if (j <= prevPos) break; // ascending bucket: the rest is even earlier
      if (y[j] >= prevY) {
        kept.push(j);
        prevPos = j;
        prevY = y[j];
        picked = true;
        break;
      }
    }
    if (!picked) throw new Error('repair: greedy invariant violated');
  }
  return kept;
}

/**
 * Pure repair analysis. Computes the largest jointly satisfiable pin set
 * without touching the caller's pins, baseline or any UI state. Returns
 * DAY_OVERFLOW when the duration prefix alone exceeds the day — then no
 * amount of unpinning can restore feasibility and no plan is produced.
 */
export function analyzeRepair(input: RepairInput): RepairResult {
  const { cues, pins } = input;
  const daySpan = input.daySpan ?? DAY_MS;
  const n = cues.length;

  // Duration prefix P[i] = sum_{k < i} duration[k]; P[n-1] is the span the
  // cues themselves need, so recovery is impossible iff it exceeds the day.
  const P = new Array<number>(n);
  P[0] = 0;
  for (let i = 1; i < n; i++) P[i] = P[i - 1] + cues[i - 1].duration;
  const need = n === 0 ? 0 : P[n - 1];
  if (need > daySpan) return { ok: false, reason: 'DAY_OVERFLOW' };
  const capY = daySpan - need; // upper box edge for y = pinStart - P[idx]

  // Map iteration is insertion-ordered; work in cueIndex order.
  const entries = [...pins].sort((a, b) => a[0] - b[0]);

  const forced: number[] = [];
  const cleanIdx: number[] = [];
  const cleanY: number[] = [];
  for (const [idx, start] of entries) {
    // Pins the solver itself would reject (bad index, non-integer start) can
    // never be kept either, so they join the forced releases.
    const y =
      Number.isInteger(idx) && idx >= 0 && idx < n && Number.isInteger(start)
        ? start - P[idx]
        : NaN;
    if (!Number.isFinite(y) || y < 0 || y > capY) {
      forced.push(idx);
    } else {
      cleanIdx.push(idx);
      cleanY.push(y);
    }
  }

  const keptPos = longestKept(cleanY);
  const keep = new Map<number, number>();
  for (const p of keptPos) {
    const idx = cleanIdx[p];
    keep.set(idx, pins.get(idx)!);
  }
  const unpin: number[] = [];
  for (const [idx] of entries) if (!keep.has(idx)) unpin.push(idx);

  return { ok: true, analysis: { keep, unpin, forced, keptCount: keep.size } };
}

// ---------------------------------------------------------------------------
// Plan identity. A generated plan is bound to the exact working state it was
// computed from: the draft object, its baseline array and the pins Map.
// Every mutation path (import, adopt, pin add/edit/remove) installs fresh
// objects, so reference equality is an exact revision check — a stale plan
// can never be applied and never modifies anything partially.
// ---------------------------------------------------------------------------

export interface RepairIdentities {
  /** Working-draft identity at generation time. */
  draft: unknown;
  /** Baseline (adopted starts) identity at generation time. */
  base: unknown;
  /** Pin-revision identity at generation time (the pins Map itself). */
  pins: unknown;
}

export interface RepairPlan extends RepairIdentities {
  analysis: RepairAnalysis;
}

export function createRepairPlan(
  analysis: RepairAnalysis,
  ids: RepairIdentities,
): RepairPlan {
  return { analysis, draft: ids.draft, base: ids.base, pins: ids.pins };
}

export function isRepairPlanCurrent(
  plan: RepairPlan,
  ids: RepairIdentities,
): boolean {
  return plan.draft === ids.draft && plan.base === ids.base && plan.pins === ids.pins;
}

export type ApplyRepairResult =
  | { applied: true; pins: Map<number, number> }
  | { applied: false; reason: 'STALE' };

/**
 * Re-verify the plan against the current identities, then — and only then —
 * hand out the replacement pin Map in one go. A stale plan applies nothing.
 */
export function applyRepairPlan(
  plan: RepairPlan,
  ids: RepairIdentities,
): ApplyRepairResult {
  if (!isRepairPlanCurrent(plan, ids)) return { applied: false, reason: 'STALE' };
  // Fresh Map: the caller swaps state wholesale and never aliases the plan.
  return { applied: true, pins: new Map(plan.analysis.keep) };
}
