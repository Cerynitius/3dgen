#!/usr/bin/env bash
# Put every bug back, one at a time, and require bugfix_test.mjs to catch it.
#
# A suite that has never been seen to fail is not evidence. Each mutation below is the
# code as WALL_REVIEW.md found it; if the suite still passes with one applied, the test
# for that fix is decoration and says so here rather than in six months.
#
#   ./mutation_bugfix.sh
set -u
cd "$(dirname "$0")"

# Restore on interrupt: a mutation left applied is a bug shipped.
restore_all() {
    for f in *.mut-orig ../../experiments/labyrinth/endless/*.mut-orig; do
        [ -e "$f" ] && mv "$f" "${f%.mut-orig}"
    done
    return 0
}
trap 'echo; echo "interrupted -- restoring"; restore_all; exit 130' INT TERM
restore_all

TESTS=bugfix_test.mjs
caught=0; missed=0

# mutate <file> <name> <expected failing test substring> <python-replacement>
mutate() {
    local file="$1" name="$2" want="$3" repl="$4"
    cp "$file" "$file.mut-orig"
    if ! python3 -c "
import io,sys
p=${file@Q}
s=io.open(p,encoding='utf-8').read()
old,new=${repl}
if s.count(old)!=1:
    sys.stderr.write('  mutation did not apply cleanly (%d hits)\n' % s.count(old)); sys.exit(2)
io.open(p,'w',encoding='utf-8').write(s.replace(old,new))
"; then
        echo "  SKIP  $name -- could not apply"
        mv "$file.mut-orig" "$file"; return
    fi

    # Time-boxed: the missing-deadline mutation makes the suite hang forever, which is
    # precisely the failure it describes, so a timeout counts as a catch rather than an
    # inconclusive run.
    local out; out=$(QUIET=1 timeout 25 node "$TESTS" 2>&1); local rc=$?
    mv "$file.mut-orig" "$file"

    if [ $rc -eq 124 ]; then
        echo "  caught  $name (the suite hung, which is the bug)"
        caught=$((caught+1)); return
    fi
    if [ $rc -eq 0 ]; then
        echo "  MISSED  $name -- suite still passes with the bug back"
        missed=$((missed+1))
    elif echo "$out" | grep -qF "$want"; then
        echo "  caught  $name"
        caught=$((caught+1))
    else
        echo "  WRONG TEST  $name -- suite failed, but not on \"$want\""
        echo "$out" | grep FAIL | sed 's/^/            /'
        missed=$((missed+1))
    fi
}

echo "reintroducing each bug in WALL_REVIEW.md:"

mutate netpolicy.js "no fetch deadline (severe 1)" \
    "aborts at the deadline" \
    "('const timer = setTimeout(() => ctl.abort(), ms);', 'const timer = setTimeout(() => {}, ms);')"

mutate netpolicy.js "no refusal cache (severe 3)" \
    "standing still costs 5 requests" \
    "('    has(key, cost = undefined) {', '    has(key, cost = undefined) { return false;')"

mutate netpolicy.js "no backoff, retry at full rate (medium 8)" \
    "backs off and then caps" \
    "('Math.min(base * 2 ** (tries - 1), cap)', 'base')"

mutate controls.js "frame-rate-dependent acceleration (medium 4)" \
    "30 to 240 Hz" \
    "('''        const blend = 1 - Math.exp(-k * dt);
        const v0x = this.vel[0], v0z = this.vel[1];
        let dx = tx * dt + (v0x - tx) * blend / k;
        let dz = tz * dt + (v0z - tz) * blend / k;''', '''        const blend = Math.min(1, k * dt);
        const v0x = this.vel[0], v0z = this.vel[1];
        let dx = (v0x + (tx - v0x) * blend) * dt;
        let dz = (v0z + (tz - v0z) * blend) * dt;''')"

mutate controls.js "no escape from inside a wall (medium 5)" \
    "a body inside solid steps out" \
    "('        if (collide && collide(this.pos[0], this.pos[2], this.radius)) {', '        if (false) {')"

mutate occupancy.js "wall map grows forever (P2)" \
    "wall map stops growing" \
    "('        const keep = (2 * half + 1) ** 2;', '        return 0;')"

mutate splats.js "sort race draws evicted tiles (P1-3)" \
    "is discarded" \
    "('            if (gen !== this.generation) return;', '')"

mutate splats.js "no MAX_TEXTURE_SIZE check (P2)" \
    "cut to fit" \
    "('        if (this.slotSize * slots > maxSplats) {', '        if (false) {')"

mutate splats.js "silent slot truncation (P0-2)" \
    "too big for its slot says so" \
    "('        if (n > room)', '        if (false)')"

mutate splats.js "clone the whole position array per tile" \
    "only the slot that changed is sent" \
    "('        if (chunkPos) { msg.base = base; msg.chunk = chunkPos; }', '        msg.positions = new Float32Array(this.capacity * 3);')"

mutate endless.js "cell index off by half a cell" \
    "cell that owns it" \
    "('    return [Math.floor((gx - 1) / 2), Math.floor((gz - 1) / 2)];', '    return [Math.floor((gx - 1.5) / 2), Math.floor((gz - 1.5) / 2)];')"

mutate ../../experiments/labyrinth/endless/tiles.py "server ring centred one tile behind" \
    "same tile the geometry does" \
    "('    return int((gx - 1) // 2) // span, int((gz - 1) // 2) // span', '    return int((gx - 1.5) // 2) // span, int((gz - 1.5) // 2) // span')"

mutate geometry.js "collision mesh is a hollow shell (P0-1)" \
    "the solid mesh seals them" \
    "('        mb.box((x0 + x1) / 2, H / 2, (z0 + z1) / 2, x1 - x0, H, z1 - z0, WALL);', '        return;')"

mutate live.js "24 slots, one splat under the worst chunk (P0-2)" \
    "sizes a slot above the worst chunk" \
    "('const SLOTS = Math.max(4, Math.floor(8_000_000 / Math.ceil(maxSplats * 1.1)));', 'const SLOTS = 24;')"

mutate live.js "collide against the splats, holes and all (P1-5)" \
    "no longer collides against the splats" \
    "('    player.update(dt, keys, collide);', '''    player.update(dt, keys, (x, z, r) => field.blocked(x, z, r));''')"

# The five below are the ones the ORIGINAL suite could not catch: it guarded those fixes
# with regexes over the source TEXT, so only a mutation that edited that exact text
# scored. Each of these puts the real defect back while leaving every string the old
# regexes matched completely untouched.

mutate splats.js "shader reads the wire quaternion vector-first (P1-4, comments intact)" \
    "vector-first, scalar-last" \
    "('    vec4 q = vec4(b.w, c.x, c.y, b.z);', '    vec4 q = vec4(b.z, b.w, c.x, c.y);')"

mutate splats.js "projection jacobian transposed (GLSL mat3 is column-major)" \
    "perspective shear" \
    "('        -(u_focal.x * cam.x) / z2, -(u_focal.y * cam.y) / z2, 0.0);', '        0.0, 0.0, 0.0);')"

mutate endless.js "cell index off by half a cell, formula text UNCHANGED" \
    "cell that owns it" \
    "('    const gx = x / CELLM(), gz = z / CELLM();', '    const gx = x / CELLM() - 0.5, gz = z / CELLM() - 0.5;')"

mutate ../../experiments/labyrinth/endless/tiles.py "server ring off by half a cell, formula text UNCHANGED" \
    "same tile the geometry does" \
    "('    gx, gz = x_m / CELL_M, z_m / CELL_M', '    gx, gz = x_m / CELL_M - 0.5, z_m / CELL_M - 0.5')"

mutate splats.js "slot cost stored as float32 against float64 challengers" \
    "cost store is float64" \
    "('        this.slotCost = new Float64Array(slots);', '        this.slotCost = new Float32Array(slots);')"

echo
echo "$caught caught, $missed missed"
[ "$missed" -eq 0 ] || exit 1
