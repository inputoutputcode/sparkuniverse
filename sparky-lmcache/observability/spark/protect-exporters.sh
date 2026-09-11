#!/usr/bin/env bash
# The benchmark scripts wipe containers between runs with a guard that only
# spares dcgm-exporter:
#
#     docker ps -a --format '{{.Names}}' | grep -v '^dcgm' | xargs -r docker rm -f
#
# That deletes the obs-* exporters. Widen the guard to '^(dcgm|obs-)'.
#
# Dry run by default - prints which files would change and how. Nothing is
# modified without --apply, and every modified file gets a .bak.
#
#   ./protect-exporters.sh            # show what would change
#   ./protect-exporters.sh --apply    # do it
set -uo pipefail
APPLY=${1:-}
PATTERN="grep -v '\^dcgm'"
REPLACE="grep -vE '^(dcgm|obs-)'"

mapfile -t FILES < <(grep -l -- "grep -v '\^dcgm'" "$HOME"/*.sh 2>/dev/null)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "no scripts in $HOME contain the narrow dcgm guard"
  echo
  echo "checking for other container-deleting patterns that would also need it:"
  grep -n -- 'docker rm -f' "$HOME"/*.sh 2>/dev/null | grep -v 'obs-' || echo "  none"
  exit 0
fi

echo "files with the narrow guard (${#FILES[@]}):"
for f in "${FILES[@]}"; do
  echo "  $f"
  grep -n -- "grep -v '\^dcgm'" "$f" | sed 's/^/      /'
done

# Anything that deletes containers WITHOUT the dcgm guard is more dangerous
# still, because it spares nothing at all.
echo
echo "container deletion without any guard (review these by hand):"
grep -ln -- 'docker rm -f' "$HOME"/*.sh 2>/dev/null \
  | grep -vxFf <(printf '%s\n' "${FILES[@]}") | sed 's/^/  /' || echo "  none"

if [ "$APPLY" != "--apply" ]; then
  echo
  echo "dry run. re-run with --apply to modify the ${#FILES[@]} file(s) above."
  exit 0
fi

for f in "${FILES[@]}"; do
  cp -p "$f" "$f.bak"
  sed -i "s|grep -v '\^dcgm'|grep -vE '^(dcgm\|obs-)'|g" "$f"
  echo "patched $f (backup at $f.bak)"
done

echo
echo "verifying:"
grep -n -- 'grep -vE' "${FILES[@]}" | sed 's/^/  /'
