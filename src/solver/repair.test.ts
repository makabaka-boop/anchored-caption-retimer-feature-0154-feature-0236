import { describe, it, expect } from 'vitest';
import { solve, solveWithSpan, DAY_MS, type Cue, type Pins } from './solve';
import {
  analyzeRepair,
  applyRepairPlan,
  createRepairPlan,
  isRepairPlanCurrent,
  type RepairAnalysis,
} from './repair';

function makeCues(durations: number[]): Cue[] {
  // start at the earliest feasible slot; repair analysis only reads durations
  const cues: Cue[] = new Array(durations.length);
  let p = 0;
  for (let i = 0; i < durations.length; i++) {
    cues[i] = { start: p, duration: durations[i], text: `c${i}` };
    p += durations[i];
  }
  return cues;
}

function prefix(durations: number[]): number[] {
  const P: number[] = new Array(durations.length);
  P[0] = 0;
  for (let i = 1; i < durations.length; i++) P[i] = P[i - 1] + durations[i - 1];
  return P;
}

function lexLess(a: number[], b: number[]): boolean {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] !== b[i]) return a[i] < b[i];
  }
  return a.length < b.length;
}

/**
 * Oracle: enumerate every kept subset, keep the feasible ones (judged by the
 * solver itself), take the maximum cardinality, and break ties by the
 * lexicographically smallest complete release list.
 */
function bruteRepair(
  cues: Cue[],
  pins: Pins,
  daySpan: number,
): { keptCount: number; unpin: number[] } {
  const entries = [...pins].sort((a, b) => a[0] - b[0]);
  const allIdx = entries.map((e) => e[0]);
  const base = cues.map((c) => c.start);
  let bestSize = -1;
  let bestUnpin: number[] | null = null;
  for (let mask = 0; mask < 1 << entries.length; mask++) {
    const kept = new Map<number, number>();
    for (let b = 0; b < entries.length; b++) {
      if (mask & (1 << b)) kept.set(entries[b][0], entries[b][1]);
    }
    if (!solveWithSpan(cues, base, kept, daySpan).ok) continue;
    const unpin = allIdx.filter((i) => !kept.has(i));
    if (
      kept.size > bestSize ||
      (kept.size === bestSize && bestUnpin !== null && lexLess(unpin, bestUnpin))
    ) {
      bestSize = kept.size;
      bestUnpin = unpin;
    }
  }
  if (bestUnpin === null) throw new Error('oracle found no feasible subset');
  return { keptCount: bestSize, unpin: bestUnpin };
}

function analyzeOrThrow(cues: Cue[], pins: Pins, daySpan?: number): RepairAnalysis {
  const r = analyzeRepair({ cues, pins, daySpan });
  if (!r.ok) throw new Error('expected a repair plan');
  return r.analysis;
}

describe('repair vs exhaustive subset enumeration (short sequences)', () => {
  // capY = daySpan - P[n-1]: config 2 has a wide box (many equal-y ties),
  // config 3 has capY = 0 (every clean pin sits exactly on the box edge).
  const configs: Array<{ durations: number[]; daySpan: number }> = [
    { durations: [2, 1, 3], daySpan: 5 },
    { durations: [1, 1, 1, 1], daySpan: 6 },
    { durations: [3, 1, 2], daySpan: 4 },
  ];

  it('matches the oracle on retention, release list, filtering and ties', () => {
    for (const { durations, daySpan } of configs) {
      const n = durations.length;
      const cues = makeCues(durations);
      const P = prefix(durations);
      const capY = daySpan - P[n - 1];
      // Pin values straddling both envelope edges in y-space.
      const yGrid = [...new Set([-1, 0, 1, capY - 1, capY, capY + 1])].sort(
        (a, b) => a - b,
      );
      const pins = new Map<number, number>();
      let checked = 0;
      const rec = (i: number): void => {
        if (i === n) {
          const expected = bruteRepair(cues, pins, daySpan);
          const a = analyzeOrThrow(cues, pins, daySpan);
          // maximum retention and the complete release list
          expect(a.keptCount).toBe(expected.keptCount);
          expect(a.unpin).toEqual(expected.unpin);
          // keep is exactly pins minus the release list, values preserved
          expect(a.keep.size).toBe(expected.keptCount);
          for (const [idx, s] of a.keep) expect(pins.get(idx)).toBe(s);
          for (const idx of a.unpin) expect(a.keep.has(idx)).toBe(false);
          // boundary filtering: forced == envelope violators, all released
          const forcedExpect = [...pins.entries()]
            .filter(([idx, s]) => s - P[idx] < 0 || s - P[idx] > capY)
            .map(([idx]) => idx)
            .sort((x, y) => x - y);
          expect(a.forced).toEqual(forcedExpect);
          for (const f of a.forced) expect(a.unpin).toContain(f);
          // the kept set is feasible for the solver itself
          expect(
            solveWithSpan(cues, cues.map((c) => c.start), a.keep, daySpan).ok,
          ).toBe(true);
          checked++;
          return;
        }
        rec(i + 1); // cue i left unpinned
        for (const y of yGrid) {
          pins.set(i, P[i] + y);
          rec(i + 1);
        }
        pins.delete(i);
      };
      rec(0);
      expect(checked).toBe((yGrid.length + 1) ** n);
    }
  });
});

describe('repair analysis unit cases', () => {
  it('releases the first pin for transformed values [5, 1, 4]', () => {
    const cues = makeCues([1, 1, 1]); // P = [0, 1, 2]
    const pins = new Map([
      [0, 5], // y = 5
      [1, 2], // y = 1
      [2, 6], // y = 4
    ]);
    const a = analyzeOrThrow(cues, pins);
    expect(a.forced).toEqual([]);
    expect(a.unpin).toEqual([0]);
    expect(a.keptCount).toBe(2);
    expect([...a.keep.entries()]).toEqual([
      [1, 2],
      [2, 6],
    ]);
  });

  it('breaks maximum-set ties by the lexicographically smallest release list', () => {
    const cues = makeCues([1, 1, 1, 1]); // P = [0, 1, 2, 3]
    // y = [1, 2, 1, 2]: maximum kept sets {0,1,3} and {0,2,3} tie at 3;
    // release lists [2] and [1] -> release [1] wins.
    const a = analyzeOrThrow(
      cues,
      new Map([
        [0, 1],
        [1, 3],
        [2, 3],
        [3, 5],
      ]),
    );
    expect(a.keptCount).toBe(3);
    expect(a.unpin).toEqual([1]);
    expect([...a.keep.keys()]).toEqual([0, 2, 3]);

    // y = [2, 1, 2, 1]: maxima {0,2}, {1,2}, {1,3}; release lists [1,3],
    // [0,3], [0,2] -> release [0,2] wins, keep {1,3}.
    const b = analyzeOrThrow(
      cues,
      new Map([
        [0, 2],
        [1, 2],
        [2, 4],
        [3, 4],
      ]),
    );
    expect(b.keptCount).toBe(2);
    expect(b.unpin).toEqual([0, 2]);
    expect([...b.keep.keys()]).toEqual([1, 3]);
  });

  it('keeps pins exactly on the envelope and releases those outside', () => {
    const cues = makeCues([2, 1, 3, 1]); // P = [0, 2, 3, 6], need 6
    const daySpan = 8; // capY = 2
    const a = analyzeOrThrow(
      cues,
      new Map([
        [0, 0], // y = 0: lower edge, keep-eligible
        [1, 4], // y = 2: upper edge, keep-eligible
        [2, 2], // y = -1: below the box, forced
        [3, 9], // y = 3: above the box, forced
      ]),
      daySpan,
    );
    expect(a.forced).toEqual([2, 3]);
    expect(a.unpin).toEqual([2, 3]);
    expect([...a.keep.keys()]).toEqual([0, 1]);
  });

  it('treats unsatisfiable pins (bad index, non-integer start) as forced', () => {
    const cues = makeCues([1, 1]);
    const pins = new Map<number, number>([
      [0, 0],
      [5, 0], // no such cue
      [1, 1.5], // non-integer start
    ]);
    const a = analyzeOrThrow(cues, pins);
    expect(a.forced).toEqual([1, 5]);
    expect(a.unpin).toEqual([1, 5]);
    expect([...a.keep.keys()]).toEqual([0]);
    // applying the plan restores solver feasibility
    expect(solve({ cues, base: [0, 1], pins: a.keep }).ok).toBe(true);
  });

  it('reports DAY_OVERFLOW when the duration prefix alone exceeds the day', () => {
    const cues = makeCues([10, 10]); // P[1] = 10 > 5
    const r = analyzeRepair({ cues, pins: new Map([[0, 0]]), daySpan: 5 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe('DAY_OVERFLOW');
  });

  it('does not mutate the input pins', () => {
    const pins = new Map([
      [0, 5],
      [1, 2],
      [2, 6],
    ]);
    const snapshot = [...pins];
    analyzeRepair({ cues: makeCues([1, 1, 1]), pins });
    expect([...pins]).toEqual(snapshot);
  });
});

describe('repair plan identity and application', () => {
  const cues = makeCues([1, 1, 1]);
  const generationPins = (): Map<number, number> =>
    new Map([
      [0, 5],
      [1, 2],
      [2, 6],
    ]);
  const mkPlan = (draft: { cues: Cue[]; base: number[] }, pins: Pins) =>
    createRepairPlan(analyzeOrThrow(cues, generationPins()), {
      draft,
      base: draft.base,
      pins,
    });

  it('applies a current plan by replacing the pin Map in one go', () => {
    const draft = { cues, base: [0, 1, 2] };
    const pins = generationPins();
    const plan = mkPlan(draft, pins);
    expect(isRepairPlanCurrent(plan, { draft, base: draft.base, pins })).toBe(true);
    const r = applyRepairPlan(plan, { draft, base: draft.base, pins });
    expect(r.applied).toBe(true);
    if (r.applied) {
      expect([...r.pins.entries()]).toEqual([
        [1, 2],
        [2, 6],
      ]);
      // fresh Map: the live Map is neither aliased nor mutated
      expect(r.pins).not.toBe(pins);
      expect([...pins.entries()]).toEqual([
        [0, 5],
        [1, 2],
        [2, 6],
      ]);
    }
  });

  it('stale apply after a pin edit reports expiry and changes nothing', () => {
    const draft = { cues, base: [0, 1, 2] };
    const pins = generationPins();
    const plan = mkPlan(draft, pins);
    // every pin edit path installs a fresh Map
    const edited = new Map(pins);
    edited.set(1, 3);
    expect(isRepairPlanCurrent(plan, { draft, base: draft.base, pins: edited })).toBe(
      false,
    );
    const r = applyRepairPlan(plan, { draft, base: draft.base, pins: edited });
    expect(r.applied).toBe(false);
    if (!r.applied) expect(r.reason).toBe('STALE');
    // no partial modification: the live pin set is byte-identical
    expect([...edited.entries()]).toEqual([
      [0, 5],
      [1, 3],
      [2, 6],
    ]);
  });

  it('adoption, import and baseline swaps all invalidate the plan', () => {
    const draft = { cues, base: [0, 1, 2] };
    const pins = generationPins();
    const plan = mkPlan(draft, pins);
    // adopt installs a new draft object with a new baseline array
    const adopted = { cues, base: [0, 1, 3] };
    expect(
      isRepairPlanCurrent(plan, { draft: adopted, base: adopted.base, pins }),
    ).toBe(false);
    // import installs a wholly new draft
    const imported = { cues: makeCues([1, 1, 1]), base: [0, 1, 2] };
    expect(
      isRepairPlanCurrent(plan, { draft: imported, base: imported.base, pins }),
    ).toBe(false);
    // a swapped baseline alone is already a different revision
    expect(isRepairPlanCurrent(plan, { draft, base: [9, 9, 9], pins })).toBe(false);
    // and every stale application is refused without side effects
    const r = applyRepairPlan(plan, { draft: adopted, base: adopted.base, pins });
    expect(r.applied).toBe(false);
  });
});

describe('repair at k = 20000 pins (adversarial)', () => {
  const N = 20_000;
  const DURATION = 4_000;
  const cues: Cue[] = Array.from({ length: N }, (_, i) => ({
    start: DURATION * i,
    duration: DURATION,
    text: 'x',
  }));
  const base = cues.map((c) => c.start);
  const need = DURATION * (N - 1); // P[n-1] = 79_996_000 <= DAY_MS
  const capY = DAY_MS - need; // 6_404_000

  it('handles a strictly decreasing y sequence in one pass', () => {
    // y[i] = capY - i: every pair conflicts, so the maximum kept set has a
    // single pin; the lex-min release keeps the last cue.
    const pins = new Map<number, number>();
    for (let i = 0; i < N; i++) pins.set(i, DURATION * i + (capY - i));
    const t0 = performance.now();
    const a = analyzeOrThrow(cues, pins);
    const elapsed = performance.now() - t0;
    expect(elapsed).toBeLessThan(2_000);
    expect(a.forced).toEqual([]);
    expect(a.keptCount).toBe(1);
    expect([...a.keep.keys()]).toEqual([N - 1]);
    expect(a.unpin.length).toBe(N - 1);
    expect(a.unpin[0]).toBe(0);
    expect(a.unpin[N - 2]).toBe(N - 2);
    // applying the plan is necessarily feasible
    const solved = solve({ cues, base, pins: a.keep });
    expect(solved.ok).toBe(true);
    if (solved.ok) expect(solved.starts[N - 1]).toBe(pins.get(N - 1));
  });

  it('resolves a 0/capY zig-zag with the lex-min release list', () => {
    // y alternates 0, capY: the maximum kept set has 10001 pins (all zeros
    // plus the final capY); the tie-break releases 1, 3, ..., 19997.
    const pins = new Map<number, number>();
    for (let i = 0; i < N; i++) {
      pins.set(i, DURATION * i + (i % 2 === 1 ? capY : 0));
    }
    const t0 = performance.now();
    const a = analyzeOrThrow(cues, pins);
    const elapsed = performance.now() - t0;
    expect(elapsed).toBeLessThan(2_000);
    expect(a.keptCount).toBe(N / 2 + 1);
    expect(a.unpin.length).toBe(N / 2 - 1);
    expect(a.unpin[0]).toBe(1);
    expect(a.unpin[a.unpin.length - 1]).toBe(N - 3);
    const solved = solve({ cues, base, pins: a.keep });
    expect(solved.ok).toBe(true);
    if (solved.ok) {
      for (const [idx, s] of a.keep) expect(solved.starts[idx]).toBe(s);
    }
  });
});
