// When a request may be made, and when it must be given up on.
//
// Both clients had the same two holes, and both were silent.
//
// No deadline. A connection the server accepts but never answers -- a wedged build, a
// half-open TCP after the laptop sleeps, a process killed mid-response -- leaves its
// promise pending forever, so the `finally` that clears the request's guard flag never
// runs. Whatever that flag guards is then stopped permanently: a tile key that stays in
// `inflight` is never requested again, which is a hole in the world that never fills;
// a `mazeInflight` that stays true freezes collision where it stood; a `ringBusy` that
// stays true ends streaming altogether. None of it logs anything.
//
// No backoff. Every failure retried at the pump's own tick, forever, at full rate: a
// failing server got hit several times a second by every candidate tile at once, and a
// tile the field refused was re-downloaded in full and thrown away again 0.25 s later
// for as long as the player stood still.
//
// Deliberately free of browser globals so it can be tested under node against a real
// socket rather than reasoned about. See bugfix_test.mjs.

/** Run `fn(signal)` with an abort deadline, and name the timeout in the error. */
export async function withDeadline(ms, label, fn) {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), ms);
    try {
        return await fn(ctl.signal);
    } catch (e) {
        if (ctl.signal.aborted) throw new Error(`${label} timed out after ${ms} ms`);
        throw e;
    } finally {
        clearTimeout(timer);
    }
}

/**
 * GET JSON under a deadline.
 *
 * The body read is inside the same deadline as the headers on purpose: a response that
 * stalls after its header block is the same hang with extra steps, and wrapping only
 * the fetch() call would catch none of it.
 */
export function getJSON(url, ms, label, fetchImpl = fetch) {
    return withDeadline(ms, label, async (signal) => {
        const r = await fetchImpl(url, { signal });
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
    });
}

/** Exponential backoff for the nth consecutive attempt, capped. */
export const backoff = (tries, base, cap) => Math.min(base * 2 ** (tries - 1), cap);

/**
 * A set of keys that are serving a wait, and why.
 *
 * A refusal records the COST it was refused at. The field turns a tile away only
 * because every resident tile is nearer, and the one thing that can change that answer
 * is the player walking closer -- so closing the distance cancels the wait outright.
 * A timer alone would have to be tuned to guess when that happened.
 */
export class Defer {
    constructor(now = () => performance.now()) {
        this.now = now;
        this.map = new Map();            // key -> { until, cost, tries }
    }

    /** Is `key` still waiting? Passing the current cost lets a closer approach cancel it. */
    has(key, cost = undefined) {
        const d = this.map.get(key);
        if (!d) return false;
        if (this.now() >= d.until) { this.map.delete(key); return false; }
        if (d.cost !== null && cost !== undefined && cost < d.cost - 0.25) {
            this.map.delete(key);
            return false;
        }
        return true;
    }

    /**
     * Hold `key` back until `waitOf(nth consecutive attempt)` has passed.
     * @returns the attempt count, so callers can log an escalating failure.
     */
    hold(key, waitOf, cost = null) {
        const tries = (this.map.get(key)?.tries ?? 0) + 1;
        this.map.set(key, { until: this.now() + waitOf(tries), cost, tries });
        return tries;
    }

    /** Forget any wait on `key` -- call this when it finally succeeds. */
    clear(key) { this.map.delete(key); }

    get size() { return this.map.size; }
}
