#!/usr/bin/env bash
# Can a peer read another node's L2 (disk), or only its L1?
#
#   ./l2-smoke.sh
#
# Two stages. The first is a control: if it fails, the second means nothing.
#
#   stage 1   send a long prompt, `lmcache kvcache clear` to empty L1 while
#             leaving disk untouched, send the same prompt again. A hit proves
#             LMCache retrieves from its own disk.
#
#   stage 2   clear L1 on both nodes, then send the same prompt again under
#             round-robin routing so it lands on the *other* node. A hit there
#             proves a peer can serve L2. A miss proves it cannot, and the disk
#             tier is local-only.
#
# The deployment sets --l2-store-policy skip_l1, which deletes each key from L1
# after storing it to L2. Together with the clear, a hit cannot have come from
# anywhere but disk. Without skip_l1 the chunks live in both tiers and every
# result is ambiguous.
#
# Why this matters: findings.md records cross-node KV as served from L1. If
# that holds, a disk tier buys Super local capacity and nothing shareable, so
# it would help a router-only arm exactly as much as a P2P arm and does not
# belong in the article's P2P story.
set -uo pipefail

ENDPOINT=${ENDPOINT:-http://localhost:8000}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
PROMPT_TOKENS=${PROMPT_TOKENS:-8000}
OUT=${OUT:-/tmp/l2-smoke}
mkdir -p "$OUT"

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

# LMCache's own counters, one scrape per node. Read these rather than the
# response body: `--enable-prompt-tokens-details` is not accepted by
# dynamo.vllm (dead-ends.md), so cached_tokens does not come back in the JSON.
#
# **Every metric, not a filtered subset.** The first version grepped for
# hit|miss|retrieve|store|p2p|l2 and reported "no p2p counters moved", which was
# very nearly recorded as a finding. The server exports exactly one p2p series,
# `lmcache_mp_active_p2p_lookup_jobs`, and the p2p_time_to_transfer histogram
# the docs describe does not exist here at all. A regex that cannot match the
# evidence produces a confident negative, which is the failure mode CLAUDE.md
# warns about: a test that cannot fail is not a test.
snap() {
  local tag=$1
  for ip in $NODES; do
    sshq "$ip" "curl -s -m 5 http://127.0.0.1:9500/metrics" 2>/dev/null \
      | grep '^lmcache_' > "$OUT/$tag-$ip.txt"
  done
}

delta() {
  local a=$1 b=$2
  for ip in $NODES; do
    echo "  --- $ip"
    join -j1 -o 0,1.2,2.2 \
      <(awk '{print $1, $2}' "$OUT/$a-$ip.txt" | sort) \
      <(awk '{print $1, $2}' "$OUT/$b-$ip.txt" | sort) 2>/dev/null \
      | awk '{d=$3-$2; if (d != 0) printf "    %-56s %+d\n", $1, d}'
  done
}

# A prompt long enough to span several 2128-token chunks, identical within a
# run so there is a prefix to reuse, and **unique between runs**.
#
# The nonce goes first so every chunk differs from every previous run's. Without
# it the second run finds both nodes already holding the chunks from the first,
# every hit is local, and the peer question can never be reached. That is what
# happened on the first attempt: both nodes ended with 1.1 GB and 36 files, so
# neither ever had to ask the other.
NONCE=${NONCE:-$(date +%s)-$RANDOM}

build_prompt() {
  python3 - "$PROMPT_TOKENS" "$NONCE" <<'PY'
import json, sys
n, nonce = int(sys.argv[1]), sys.argv[2]
# `item000123` is several tokens, not one. The first run asked for 8,000 and
# built 76,608, so this is a floor rather than a target -- the response's
# usage.prompt_tokens is what the script reports.
words = [f"item{i:06d}" for i in range(int(n * 1.4))]
print(json.dumps({
    "model": "nemotron",
    "prompt": f"Run {nonce}. Catalogue: " + " ".join(words) + "\nSummarise.",
    "max_tokens": 8,
    "temperature": 0.0,
}))
PY
}

send() {
  curl -s -m 300 -X POST "$ENDPOINT/v1/completions" \
    -H 'Content-Type: application/json' -d @"$OUT/prompt.json" \
    -o "$OUT/resp-$1.json" -w '%{http_code}'
}

# `lmcache kvcache clear --url URL` wants the MP **HTTP** endpoint, which is
# --http-port 9500 here. The first version passed --host/--port aimed at the ZMQ
# port 5555, both stages ran against a full L1, and the results meant nothing.
clear_l1() {
  for ip in $NODES; do
    sshq "$ip" "sudo -n docker run --rm --network host \
      dynamo-vllm-lmcache:1.3.0-lmc052-arm64 \
      lmcache kvcache clear --url http://127.0.0.1:9500" >/dev/null 2>&1 \
      && echo "  L1 cleared on $ip" \
      || echo "  !! kvcache clear FAILED on $ip -- stages below are invalid"
  done
}

# usage.prompt_tokens from the response, so the log says what was actually sent
# rather than what was asked for.
tokens_of() {
  python3 -c "
import json,sys
try:
    d=json.load(open('$1'))
    u=d.get('usage') or {}
    print(f\"prompt {u.get('prompt_tokens','?')} tok, completion {u.get('completion_tokens','?')}\")
except Exception as e:
    print('no usage in response:', e)"
}

SETTLE=${SETTLE:-30}

echo "run nonce $NONCE   (unique prompt, so no chunk from an earlier run can be reused)"
build_prompt > "$OUT/prompt.json"
wc -c < "$OUT/prompt.json" | xargs echo "  prompt bytes:"

curl -sf -m 10 "$ENDPOINT/v1/models" >/dev/null || {
  echo "!! endpoint not serving at $ENDPOINT"; exit 1; }

echo
echo "=== send 1: nothing anywhere holds this prompt. One node prefills and stores ==="
snap t0
echo "  http $(send warm)   $(tokens_of "$OUT/resp-warm.json")"
sleep "$SETTLE"
snap t1
delta t0 t1
echo "  disk after send 1:"
for ip in $NODES; do
  echo -n "    $ip  "
  sshq "$ip" 'du -sh /mnt/lmcache-l2 2>/dev/null | cut -f1; find /mnt/lmcache-l2 -name "*.data" | wc -l' | tr '\n' ' '
  echo
done

echo
echo "=== send 2: L1 cleared everywhere, round-robin sends it to the other node ==="
echo "    that node has no local copy of this nonce, so a hit can only be a peer"
clear_l1
sleep 5
snap t2
echo "  http $(send stage2)   $(tokens_of "$OUT/resp-stage2.json")"
sleep "$SETTLE"
snap t3
delta t2 t3

echo
echo "=== disk contents per node ==="
for ip in $NODES; do
  echo "  --- $ip"
  sshq "$ip" 'du -sh /mnt/lmcache-l2 2>/dev/null; find /mnt/lmcache-l2 -name "*.data" | wc -l' \
    | sed 's/^/    /'
done

echo
echo "Read it this way. The node whose counters moved on send 2 is the one that"
echo "answered it; the other is the one that wrote the chunks on send 1."
echo
echo "  PEERS CAN READ L2"
echo "    send 2 shows l2_prefetch_hit_chunks rising on the answering node"
echo "    while only the other node holds .data files for this nonce."
echo
echo "  PEERS CANNOT READ L2"
echo "    send 2 shows lookup_requested_tokens with little or no hit, and"
echo "    l2_store counters rising as the answering node prefills and writes"
echo "    its own copy. Disk is then local only, and would help a router-only"
echo "    arm exactly as much as a P2P arm."
echo
echo "  INCONCLUSIVE"
echo "    both nodes hold files for this nonce, or neither set of counters"
echo "    moved. Round-robin sent both sends to the same node, or the settle"
echo "    was too short. Rerun with SETTLE=60."
echo
echo "Then redeploy with COORD_L2_EVENTS=1 and run this again. That flag is"
echo "the only documented path to fleet-wide L2 discovery, so a different"
echo "result there is the whole answer."
echo
echo "raw scrapes in $OUT"
