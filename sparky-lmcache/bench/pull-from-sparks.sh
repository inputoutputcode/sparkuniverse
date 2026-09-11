#!/usr/bin/env bash
# Every measurement script currently lives only in the home directory of one or
# both Sparks. Pull them into the repo so the results are reproducible.
#
# One glob per rsync invocation: a multi-line quoted file list is passed to the
# remote shell as a single filename and fails with "No such file or directory".
set -uo pipefail
DEST=$(cd "$(dirname "$0")" && pwd)
# name=ip pairs. IPs, not hostnames: spark-a / spark-a resolve only in
# ~/.ssh/config, and an unresolvable name fails in ways that look like an
# empty home directory.
NODES=${NODES:-"spark-a=10.0.0.11 spark-b=10.0.0.12"}

for spec in $NODES; do
  name=${spec%%=*}; ip=${spec#*=}
  echo "=== $name ($ip)"
  ssh -o ConnectTimeout=5 "$ip" true 2>/dev/null || { echo "  unreachable, skipping"; continue; }
  mkdir -p "$DEST/scripts/$name"
  for pat in '~/*.sh' '~/*.py'; do
    rsync -av --no-motd "$ip:$pat" "$DEST/scripts/$name/" 2>&1 \
      | grep -vE '^(sending|receiving|sent |total size)' || true
  done
done

chmod +x "$DEST/scripts"/*/*.sh 2>/dev/null || true

echo
echo "=== pulled ==="
find "$DEST/scripts" -type f | sort
echo
echo "=== on the Sparks but not pulled ==="
for spec in $NODES; do
  name=${spec%%=*}; ip=${spec#*=}
  ssh -o ConnectTimeout=5 "$ip" 'ls ~/*.sh ~/*.py 2>/dev/null | xargs -n1 basename' \
    2>/dev/null | sort > "/tmp/remote.$name"
  ls "$DEST/scripts/$name" 2>/dev/null | sort > "/tmp/local.$name"
  if [ -s "/tmp/remote.$name" ]; then
    comm -23 "/tmp/remote.$name" "/tmp/local.$name" | sed "s/^/  $name: /"
  else
    echo "  $name: could not list remote"
  fi
done

echo
echo "Review before committing. Several carry hardcoded paths, model names and"
echo "node addresses that belong in variables."
