#!/usr/bin/env bash
# Hermetic checks for install.sh: stage bookkeeping, probes, skip logic,
# dry-run, idempotent alias. Run: bash tests/shell/test_install_sh.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO/install.sh"
pass=0; fail=0
t_ok()   { pass=$((pass+1)); echo "  ok   $1"; }
t_bad()  { fail=$((fail+1)); echo "  FAIL $1"; }
check() { if eval "$2"; then t_ok "$1"; else t_bad "$1"; fi; }

echo "== install.sh exists and is executable"
check "script present" '[ -x "$SCRIPT" ]'
[ -x "$SCRIPT" ] || { echo "$fail failure(s)"; exit 1; }

echo "== library mode exposes functions"
INSTALL_SH_LIB=1 source "$SCRIPT"
check "render_bar renders width-30 bar with counter" '[ "$(render_bar 3 10)" = "[#########---------------------]  3/10" ]'
check "render_bar full"                               '[ "$(render_bar 10 10)" = "[##############################] 10/10" ]'
check "variant_from_branch cpu"  '[ "$(variant_from_branch cpu)" = cpu ]'
check "variant_from_branch main" '[ "$(variant_from_branch main)" = gpu ]'
check "variant_from_branch other → gpu" '[ "$(variant_from_branch feature/x)" = gpu ]'

echo "== stage tables"
select_stages gpu
check "gpu stage list starts with prereq" '[ "${STAGE_IDS[0]}" = prereq ]'
check "gpu stage list has engines"   'printf "%s\n" "${STAGE_IDS[@]}" | grep -qx engines'
check "gpu stage list has multical"  'printf "%s\n" "${STAGE_IDS[@]}" | grep -qx multical'
select_stages cpu
check "cpu stage list has paths"     'printf "%s\n" "${STAGE_IDS[@]}" | grep -qx paths'
check "cpu stage list has no engines" '! printf "%s\n" "${STAGE_IDS[@]}" | grep -qx engines'
check "every stage has a title, probe and run fn" 'for i in "${!STAGE_IDS[@]}"; do [ -n "${STAGE_TITLES[$i]}" ] && declare -F "probe_${STAGE_IDS[$i]}" >/dev/null && declare -F "run_${STAGE_IDS[$i]}" >/dev/null || exit 1; done'

echo "== skip / only logic"
SKIP_LIST=(alias comms); ONLY_STAGE=""
check "skipped stage is skipped"     'should_skip alias'
check "other stage not skipped"      '! should_skip env'
SKIP_LIST=(); ONLY_STAGE=verify
check "--only keeps that stage"      '! should_skip verify'
check "--only skips the rest"        'should_skip env'
SKIP_LIST=(); ONLY_STAGE=""

echo "== alias probe + run are idempotent (fake HOME)"
FAKE_HOME="$(mktemp -d)"; touch "$FAKE_HOME/.bashrc"
HOME="$FAKE_HOME" VARIANT=gpu
check "alias absent → probe fails"   '! probe_alias'
DRY=0; run_alias >/dev/null
check "alias added"                  'grep -q "^alias 3d=" "$FAKE_HOME/.bashrc"'
check "alias present → probe passes" 'probe_alias'
run_alias >/dev/null
check "second run does not duplicate" '[ "$(grep -c "^alias 3d=" "$FAKE_HOME/.bashrc")" = 1 ]'
VARIANT=cpu
check "cpu alias absent → probe fails" '! probe_alias'
run_alias >/dev/null
check "cpu alias added"              'grep -q "^alias 3d_cpu=" "$FAKE_HOME/.bashrc"'

echo "== dry-run never writes"
FAKE_HOME2="$(mktemp -d)"; touch "$FAKE_HOME2/.bashrc"
out="$(HOME="$FAKE_HOME2" bash "$SCRIPT" gpu --dry-run --only alias 2>&1)"
check "dry-run says what it would do"  'echo "$out" | grep -qi "would"'
check "dry-run left .bashrc untouched" '! grep -q "alias 3d=" "$FAKE_HOME2/.bashrc"'
check "dry-run exit 0"                 'HOME="$FAKE_HOME2" bash "$SCRIPT" gpu --dry-run --only alias >/dev/null 2>&1'

echo "== --list"
check "--list prints stage ids"        'bash "$SCRIPT" cpu --list | grep -q "^  paths"'

rm -rf "$FAKE_HOME" "$FAKE_HOME2"
echo; echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
