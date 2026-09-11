# Deploying the aggregated two-node scenario

`nano-agg-p2p.yaml` is the first full scenario: Nemotron 3 Nano aggregated
across both Sparks, KV-aware routing, LMCache sharing cache between the nodes
over RoCE.

Every constant in it was measured rather than chosen. The derivations are in
[`../docs/findings.md`](../docs/findings.md).

---

## 1. Get the image into containerd

K3s uses containerd, not dockerd, and they keep separate image stores. An image
built with `docker build` is invisible to pods no matter which node it is on.
Without this step you get `ImagePullBackOff` on both nodes, because Kubernetes
tries to pull `dynamo-vllm-lmcache:...` from Docker Hub.

```bash
for ip in 10.0.0.11 10.0.0.12; do
  ssh -t $ip 'docker save dynamo-vllm-lmcache:1.3.0-lmc052-arm64 \
    | sudo k3s ctr images import -
    sudo k3s ctr images ls | grep lmcache'
done
```

About 23 GB per node through a pipe, so give it a few minutes. `imagePullPolicy:
IfNotPresent` in the manifest keeps Kubernetes from trying to pull it anyway.

If you end up iterating on the image, stand up a registry on the fabric instead
and skip this per-node step entirely.

---

## 2. Check the API version your operator serves

The manifest declares `nvidia.com/v1beta1`. Your platform chart is 1.2.1, and
some example directories in the Dynamo repo carry both a `v1beta1/` variant and
an older one, which suggests both may be served.

```bash
ssh 10.0.0.11 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
k3s kubectl get crd dynamographdeployments.nvidia.com \
  -o jsonpath="{range .spec.versions[*]}{.name} served={.served} storage={.storage}{\"\n\"}{end}"'
```

If `v1beta1` is not served, change the `apiVersion` to whichever is, and check
whether the schema uses `spec.components` (a list) or `spec.services` (a map).
Your own `~/demos/dynamo-tests/qwen3-first-model.yaml` is the reference, since
that one was accepted by this operator.

---

## 3. Namespace

The manifest uses `dynamo-system`, where the platform already runs. If you
deploy elsewhere, the namespace needs the KAI label or pods sit
`SchedulingGated` forever — KAI's pod-grouper filters by namespace label.

```bash
ssh 10.0.0.11 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
k3s kubectl label namespace dynamo-system kai.scheduler/enabled=true --overwrite'
```

---

## 4. Apply

```bash
scp deploy/nano-agg-p2p.yaml 10.0.0.11:~/
ssh 10.0.0.11 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
k3s kubectl apply -f ~/nano-agg-p2p.yaml
k3s kubectl -n dynamo-system get pods -w'
```

First boot loads 19 GB of weights per node. Expect a few minutes.

---

## 5. Verify, in this order

**Both workers landed on different nodes.** Each requests one GPU and each node
has exactly one, so the scheduler has no choice. Check anyway, because if both
were on one Spark there would be no peer and every metric would still look
plausible.

Do not add `topologySpreadConstraints` to force it. Grove owns placement and
its webhook rejects them: `spec.template.cliques[1].spec.podSpec.spec.
topologySpreadConstraints: must not be set`.

```bash
k3s kubectl -n dynamo-system get pods -o wide | grep VllmDecodeWorker
```

**Both registered with the coordinator, and each attached a peer backend.**

```bash
k3s kubectl -n dynamo-system logs -l nvidia.com/dynamo-component-name=VllmDecodeWorker \
  --tail=200 | grep -iE "Registered with coordinator|Added L2 adapter"
```

Two `Registered with coordinator as spark-...` lines and two `Added L2 adapter
0 (p2p)` lines. No `--l2-adapter` flag is needed, the server attaches it once a
coordinator URL and an advertise URL are both set.

**The frontend serves.**

If `/v1/models` comes back with an empty `data` list while the workers look
healthy, the frontend is almost certainly failing to resolve the model, not
failing to find the workers. Read its log before touching anything network:

```bash
FE=$(k3s kubectl -n dynamo-system get pods -o name | grep frontend)
k3s kubectl -n dynamo-system logs $FE | grep -iE "error|warn" | head
```

`hub::from_hf(/models/...): Is this a valid HuggingFace ID?` means the frontend
has no `/models` mount. Workers register a model card carrying their local
`source_path`, and the frontend resolves it to read the tokenizer and config —
weights are skipped, note `ignore_weights is set to true`. With no mount it
treats the path as a HuggingFace repo id, 404s, and drops the model. Discovery
succeeded; you will see `Snapshot (seq=0): 2 instances, added=[...]` just above.
The manifest mounts `/models` into the frontend for this reason.

```bash
k3s kubectl -n dynamo-system port-forward svc/nano-agg-p2p-frontend 8000:8000 &
curl -s localhost:8000/v1/models | jq .
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"nemotron","prompt":"hello","max_tokens":16}' | jq -r '.choices[0].text'
```

**KV crosses the wire.** This is the one that matters. Send the same long
prompt twice and watch for a worker reporting L2 hits.

```bash
k3s kubectl -n dynamo-system logs -l nvidia.com/dynamo-component-name=VllmDecodeWorker \
  --tail=400 | grep -iE "retained keys"
```

`(x L1, y L2)` with **y > 0** means a worker served the request from its peer.
All zeros in the L2 column means the cache is working locally and P2P is not,
which looks identical in every throughput metric.

---

## 6. Standing it down, and back up

The bare-docker scripts in `bench/` — `p2p-validate.sh`,
`cache-correctness.sh`, `l2-bytes-per-token.sh`,
`mamba-align-prefix-cache.sh` — start their own vLLM with `--gpus all`. They
cannot coexist with the deployment: each Spark has one GPU and the workers
hold it. This has bitten twice, both times as an engine dying in
`init_device()` with a traceback that says nothing about memory.

Down:

```bash
ssh 10.0.0.11 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
k3s kubectl -n dynamo-system delete dgd nano-agg-p2p --ignore-not-found
k3s kubectl -n dynamo-system delete deploy lmcache-coordinator --ignore-not-found
k3s kubectl -n dynamo-system wait --for=delete pod \
  -l nvidia.com/dynamo-component=VllmDecodeWorker --timeout=180s 2>/dev/null
nvidia-smi --query-compute-apps=pid,used_memory --format=csv'
```

The `wait --for=delete` is the point. Deleting the DGD returns immediately
while the engines take tens of seconds to release GPU memory, and the next
thing to start will fail against the remainder. `nvidia-smi` must come back
with no process above ~1 GB before anything else claims the GPU — on GB10 use
`--query-compute-apps`, because `--query-gpu=memory.used` returns `[N/A]` on
unified memory.

The coordinator uses no GPU, but leaving it up while the workers are gone
produces registration noise that looks like a fault later.

Back up:

```bash
ssh 10.0.0.11 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
k3s kubectl apply -f ~/nano-agg-p2p.yaml
sleep 150
k3s kubectl -n dynamo-system get pods -o wide | grep -E "decodeworker|frontend"'
```

About 130 s to reload 19 GB of weights per node. Re-verify with the checks in
part 5 before trusting a run: the workers register with the coordinator and
attach a peer backend on every restart, and a silent failure there leaves a
deployment that serves correctly with no P2P at all.

---

## 7. Benchmark

```bash
ARM=p2p MODEL=nano ENDPOINT=http://10.0.0.11:8000 \
  TRACE=traces/data/cc-weka-s10-128k.jsonl CONCURRENCY=5 \
  ../bench/run-scenario.sh
```

Artifacts land in `runs/<run_id>/` with a `config.json`, the AIPerf export and
a `summary.json`. The run registers itself in Prometheus, so every dashboard
series can be attributed to it.

**Measuring reuse under Dynamo is not the same as standalone.**
`--enable-prompt-tokens-details` is a `vllm serve` flag and `dynamo.vllm`
rejects it, so `usage.prompt_tokens_details.cached_tokens` may be absent from
responses. Check first:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"nemotron","prompt":"hello","max_tokens":4}' | jq .usage
```

If `prompt_tokens_details` is missing, fall back to counters that do not depend
on the response payload:

- `lmcache_mp_l1_usage_ratio` and the retained-keys log lines, for L1 against
  L2 hits per node
- `vllm:prefix_cache_hits_total` over `vllm:prefix_cache_queries_total`, for
  device-side reuse

These are the three-way consistency check from the observability plan, and
disagreement between them is informative rather than a nuisance.

For the router-only arm, redeploy with the `--kv-transfer-config` argument and
the LMCache environment removed, so vLLM runs with device prefix caching alone.
For the single-node arm, set `replicas: 1`.
