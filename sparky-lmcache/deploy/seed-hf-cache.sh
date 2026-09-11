#!/usr/bin/env bash
# Build a HuggingFace cache entry from a model directory already on disk,
# without downloading the weights.
#
# Why this exists:
#   Dynamo's frontend is the preprocessor. The worker's model card says
#   "model_input": "Tokens", so the frontend applies the chat template and
#   tokenizes before anything reaches vLLM. It needs tokenizer.json and
#   config.json to do that.
#
#   The card carries whatever was passed to --model. Give it a filesystem path
#   and the frontend, which has no such path, treats it as a repo id:
#     https://huggingface.co/api/models//models/nemotron-3-nano-30b-nvfp4/...
#                                       ^^ path concatenated onto the API root
#   404, and the model is dropped from /v1/models while discovery succeeds.
#
#   Mounting the directory into the frontend is not possible. The operator's
#   frontend defaulter rewrites `type: frontend` components and drops volumes,
#   volumeMounts and compilationCache. Verified against the stored resource:
#   the worker keeps all four of its volumes, the frontend keeps none.
#
#   So the card gets the repo id. The frontend then fetches ~18 MB of small
#   files from the hub, which is the path that already worked for Qwen. The
#   worker must not download 19 GB of weights to match, hence this script:
#   it fabricates the cache layout around the files you already have.
#
# Run on BOTH Sparks. Idempotent.
set -Eeuo pipefail

REPO_ID=${REPO_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
SRC=${SRC:-$HOME/models/nemotron-3-nano-30b-nvfp4}
# Under $HOME on purpose: hard links cannot cross filesystems, and this keeps
# the cache on the same one as the model.
HF_CACHE=${HF_CACHE:-$HOME/hf-cache}

[ -d "$SRC" ] || { echo "!! no such model directory: $SRC"; exit 1; }
[ -s "$SRC/tokenizer.json" ] || { echo "!! $SRC has no tokenizer.json"; exit 1; }

# models--org--name, the hub's own directory convention
DIR="models--${REPO_ID//\//--}"
BASE="$HF_CACHE/hub/$DIR"

# The real commit sha, so the cache is indistinguishable from a downloaded one
# and stays valid if HF_HUB_OFFLINE is ever unset. A fabricated sha works
# offline but diverges the moment anything revalidates.
SHA=$(curl -sf "https://huggingface.co/api/models/$REPO_ID/revision/main" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null || true)
if [ -z "$SHA" ]; then
  echo "!! could not read the commit sha for $REPO_ID"
  echo "   check the id resolves: curl -s -o /dev/null -w '%{http_code}\n' \\"
  echo "     https://huggingface.co/api/models/$REPO_ID/revision/main"
  exit 1
fi
echo "repo $REPO_ID"
echo "sha  $SHA"

SNAP="$BASE/snapshots/$SHA"
mkdir -p "$SNAP" "$BASE/refs" "$BASE/blobs"
echo -n "$SHA" > "$BASE/refs/main"

# Hard links, not copies: same inode, so 19 GB of weights costs no extra disk
# and no read. Not symlinks either, because a symlink's target path would have
# to exist inside the container too, and this way the mount is self-contained.
n=0
for f in "$SRC"/*; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  if [ ! -e "$SNAP/$b" ]; then
    ln "$f" "$SNAP/$b" 2>/dev/null || cp "$f" "$SNAP/$b"
  fi
  n=$((n+1))
done

chmod -R a+rX "$HF_CACHE"
echo "linked $n files into $SNAP"

# The five files the frontend actually reads. A missing tokenizer.json here is
# the difference between a working deployment and an empty /v1/models.
for want in config.json tokenizer.json tokenizer_config.json \
            generation_config.json chat_template.jinja; do
  [ -e "$SNAP/$want" ] && echo "  ok   $want" || echo "  MISS $want"
done

du -sh --apparent-size "$SNAP" 2>/dev/null | sed 's/^/apparent /'
du -sh "$HF_CACHE" | sed 's/^/on disk  /'
