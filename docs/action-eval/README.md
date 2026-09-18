# DreamZero LIBERO evaluation handoff

This page defines the planned use of RLinf's DreamZero LIBERO policy in action-eval. **Execution is paused; main is the maintained evaluation/default branch.** Existing upstream installation examples are references, not authorization to modify the shared server.

## Fixed sources and candidate

Audited RLinf commit: `717dfb2cb9e59927fb9f5ea207ab63f43b119f1d`. Its installer references DreamZero `ab790c198fbce33503358efbbd4187ce9a89adf3`; preserve that lineage without resetting existing model-fork history.

Candidate: RLinf DreamZero WAN2.2-5B LIBERO-SFT. This is not original DROID14B. The Step18000 URL redirects to Step26000; the previously inspected Step26000 HF revision is `7416908d4e12c7a9fb3b7eb4fde7408fd2559e3c`, not a weight-file hash. Revalidate identity before download. The recipe points at a four-suite dataset, but actual released-checkpoint training coverage needs stronger evidence.

## Preferred future integration

Review `evaluations/libero/libero_spatial_dreamzero_eval.yaml` (native HuggingFace backend), `rlinf/models/embodiment/dreamzero/` including patches, and `rlinf/data/datasets/dreamzero/`. Preserve native transforms and temporal cache reset; do not copy DROID mappings into LIBERO. Avoid introducing SGLang until the simpler native path has been evaluated.

The upstream single-GPU SGLang configuration establishes a candidate deployment shape, not a measured native-stack H200 result. Start with one environment on authorized GPU5 only, never the default all-GPU placement or 128 environments without profiling.

## Dependency and isolation blockers

Current RLinf manifest uses a newer Torch stack than the existing DreamZero environment; DreamZero requirements also request TorchCodec0.2 while other RLinf constraints request >=0.11,<0.12. PEFT, Diffusers and Transformers differ. Preserve existing manifests/locks as baselines, document the minimal compatible changes and generate a complete lock on server staging. Do not claim unchanged dependencies already work.

Do not run requirements/install.sh blindly: LIBERO installation can reset shared ~/.libero; separate venv alone does not isolate this. Protect active 14B processes/environments, existing Ray instances, ports and compilation caches. No installation, model download or GPU run was performed for RLinf in this preparation.

## Ordered gates

1. Review exact environment lineage and minimal version differences.
2. Pin checkpoint/metadata/tokenizer and hash assets; verify training-coverage evidence.
3. Implement and test native-policy adapter parity and reset behavior.
4. Validate four-suite single-GPU smoke, renderer stability and measured wall time.
5. Run the approved 2,000-episode manifest only within an explicit budget, preserving errors and attempts.
6. Export complete per-task/per-suite counts and provenance; missing coverage cannot yield final SR.

Follow [the authoritative two-model handoff](https://github.com/winbeau/action-eval/blob/main/docs/plan/00-current-handoff.md) and [this fork's constraints](../../AGENTS.md). Future evidence records need status, timestamp/timezone, code/lock/checkpoint hashes, exact command, exit code, artifact path and limitations. All integration stages above remain unexecuted.
