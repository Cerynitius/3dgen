#!/bin/bash
# Prove the selftest has teeth.
#
# A suite that passes tells you nothing until you have watched it fail. Each mutation
# below reintroduces a specific bug this design is exposed to; the selftest must catch
# every one of them, and must name the property that broke.
#
#   bash mutation_test.sh
set -u
SRC=/home/zrx/HY-World-2.0/experiments/labyrinth/endless
WORK=/tmp/claude-1000/-home-zrx/2aba1414-6f9a-4d62-9b39-12a970e01484/scratchpad/mut
rc_all=0

run_mutation() {
    local name="$1" sed_expr="$2" expect="$3"
    rm -rf "$WORK"; mkdir -p "$WORK"
    cp "$SRC"/infinite_maze.py "$SRC"/selftest_maze.py "$WORK"/
    sed -i "$sed_expr" "$WORK"/infinite_maze.py
    if ! grep -q . "$WORK"/infinite_maze.py; then echo "  !! mutation emptied the file"; return; fi
    out=$(cd "$WORK" && timeout 300 python selftest_maze.py 2>&1)
    if echo "$out" | grep -q "ALL PASS"; then
        echo "  ESCAPED  $name  -- selftest still passed, the bug is invisible to it"
        rc_all=1
    else
        caught=$(echo "$out" | grep '^  FAIL' | sed 's/^  FAIL  //' | head -3 | paste -sd'; ')
        if [ -z "$caught" ]; then caught="crashed: $(echo "$out" | tail -1 | cut -c1-70)"; fi
        echo "  caught   $name"
        echo "           -> $caught"
        if ! echo "$out" | grep -q "$expect"; then
            echo "           !! expected the '$expect' property to break, it did not"
            rc_all=1
        fi
    fi
}

echo "mutation tests: reintroduce each bug, the selftest must catch it"
echo

run_mutation "west border reads the wrong edge id (gates disagree across chunks)" \
    's|gates_v(seed, self.ch - 1, self.rh, self.k)|gates_v(seed, self.ch, self.rh, self.k)|' \
    "wall symmetry"

run_mutation "occupancy lets doorways punch through the skirt (rays escape)" \
    's|if 0 < nr_ < 2 \* rows and 0 < nc_ < 2 \* cols:|if -1 < nr_ < 2 * rows + 1 and -1 < nc_ < 2 * cols + 1:|' \
    "skirt"

run_mutation "torch halo dropped (placement depends on which box you ask for)" \
    's|halo = int(np.ceil(self.spacing / 2.0)) + 1|halo = 0|' \
    "torch"

run_mutation "floor division replaced by truncation (breaks negative coordinates)" \
    's|^    return a // b.*$|    return int(a / b)|' \
    "reachable\|wall symmetry"

run_mutation "chunk hash ignores its coordinates (every chunk identical)" \
    's|random.Random(_hash(seed, _D_CHUNK, ch, rh))|random.Random(_hash(seed, _D_CHUNK))|' \
    "different"

# Expected to escape, and measured rather than assumed: _grow's backtracker already
# spans every cell (rooms keep a one-cell margin, so the non-room region stays
# connected around them), and _repair was needed on 0/400 chunks at every setting from
# k=8/6 rooms to k=24/40 rooms. Removing dead code changes no observable behaviour, so
# no property-based test can catch it. _repair is insurance for settings outside those
# defaults, and selftest_maze.py proves it works by sealing a cell on purpose.
echo "  (expected escape) internal repair removed -- dead code at current settings,"
echo "                    covered directly by the sealed-cell test instead"

echo
if [ $rc_all -eq 0 ]; then echo "ALL MUTATIONS CAUGHT"; else echo "SOME MUTATIONS ESCAPED"; fi
exit $rc_all
