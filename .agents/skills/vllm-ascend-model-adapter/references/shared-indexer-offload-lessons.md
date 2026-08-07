# Shared Indexer and DSA Offload Review

Use this checklist for DSA sparse attention models that share top-K selection
across layers or offload resident MLA KV blocks.

## 1. Separate two different sharing modes

- **Checkpoint-declared sharing**: `indexer_types[layer] == "shared"`; the layer
  has no local Indexer weights/cache and reuses a preceding full layer's top-K.
- **Runtime IndexCache**: `skip_topk=True`, but the layer may still own a local
  Indexer. This is an optimization policy, not proof of checkpoint sharing.

Never route every `skip_topk` layer through the checkpoint-shared path. Build an
explicit support matrix over `has_indexer`, `skip_topk`, and the current layer's
declared `indexer_types` value. Reject unsupported combinations before serving.

## 2. Validate topology at startup

For an explicit `indexer_types` sequence:

1. Accept only `full` and `shared` strings.
2. Require one entry per transformer layer.
3. Require every shared layer to have a preceding full source.
4. Emit Indexer KV specs only for full layers.
5. Compare the exact emitted Indexer layer-index set with the declared full set.
6. Require Indexer layer indices to be a subset of resident MLA layer indices,
   even when the two sets have equal cardinality.

Use transformer layer indices rather than cache-name string equality because the
resident and Indexer modules have different suffixes.

## 3. Preserve data-plane semantics

For each full layer, run LIDU and retain caller-owned outputs until its shared
followers finish:

- token indices;
- destination resident slots;
- miss counts;
- tail metadata.

For each shared layer:

- do not run LIDU;
- reuse the source full layer's complete selection output;
- run KSC against the shared layer's own resident cache and DRAM arena;
- run sparse attention and full-block dump for the shared layer itself.

Reset the source marker at every eager preparation, graph execution view, and
graph capture. Ensure full and shared operations are captured in source-before-
consumer order. Preserve native prefill/index-buffer behavior separately from
the offloaded decode path.

## 4. Keep unrelated paths unchanged

- Preserve the previous prefetch/event boundary for full Indexer layers; add a
  separate boundary only for shared layers that skip Indexer preprocessing.
- Treat all-full models without IndexCache as a regression path.
- For all-full runtime IndexCache, either implement source mapping explicitly or
  retain the existing fail-fast. Do not allow binding and then crash in decode.
- Recheck eager, FULL graph, PAD rows, chunked prefill, and continuous batching.

## 5. Do not partially scaffold quantized Indexer offload

A C8/FP8 Indexer path is complete only when all of these agree:

1. device-specific K and scale dtypes;
2. page-size accounting;
3. raw allocation and independent K/scale reshape;
4. cache binding type;
5. quantized Indexer writes;
6. prefill Lightning Indexer inputs;
7. decode query quantization and scale transport;
8. stateful quantized LIDU operator ABI;
9. eager and graph validation with real weights.

Until all nine exist, fail at configuration/spec construction. Do not add a spec
or runtime dispatch that would become reachable by removing one guard.

## 6. Minimum regression matrix

Add tests for:

- full layer: local Indexer, `skip_topk=False`;
- declared shared layer: no local Indexer, `skip_topk=True`;
- runtime IndexCache: local Indexer, `skip_topk=True` (supported or early reject);
- missing Indexer without a declared shared topology (reject);
- shared layer before any full source (reject);
- declared full set versus emitted Indexer set mismatch;
- equal-size but misaligned Indexer/resident layer sets;
- multiple shared followers and transition to the next full source;
- eager and graph source freshness;
- C8 option fail-fast until the complete quantized path lands.

Run the focused tests, `git diff --check`, and the repository formatting command.
Do not claim model support without real-weight inference and non-empty output.
