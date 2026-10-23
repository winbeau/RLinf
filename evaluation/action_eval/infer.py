"""action-eval adapter for the RLinf DreamZero WAN2.2-5B LIBERO-SFT checkpoint.

This is the only file action-eval needs from this fork. It is deliberately thin: it
wires the platform's observation contract onto RLinf's own DreamZero build and
inference path and adds no modelling code of its own.

Which checkpoint this is (and is not)
-------------------------------------

This adapter evaluates ``RLinf/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000``: a
WAN2.2-TI2V-5B (DiT) flow-matching policy fine-tuned on LIBERO. It is **not** the
DreamZero-DROID 14B model, and a number produced here must never be presented as one.
The two differ in embodiment, action space and training data.

Where each responsibility lives
-------------------------------

=============================================  ==================================
concern                                        owner
=============================================  ==================================
image flip                                     action-eval ``libero/backend.py``
center crop, resize, view concat, text prompt  RLinf ``data_transforms``
state/action q99 normalisation and inverse     RLinf ``data_transforms``
gripper binarisation                           RLinf ``predict_action_batch``
which GPU, timeouts, episode loop, success     action-eval
=============================================  ==================================

The adapter therefore applies **no** preprocessing of its own. Two mistakes in
particular would be silent rather than loud: flipping the images again (the benchmark
already applied the ``[::-1, ::-1]`` that RLinf's own LIBERO env applies), and wrapping
the instruction in the multi-view prompt template again (RLinf's collator does that
from the bare instruction).

Injected paths
--------------

The runner injects three fully resolved paths, so this file never guesses how a
relative path is meant:

``model_root``    the RLinf checkout (``policy.repo_root``)
``model_config``  the RLinf model architecture YAML (``policy.model_config``)
``checkpoint``    the DreamZero checkpoint directory (``policy.checkpoint``)

Options (``policy.options``), all optional except where noted:

``tokenizer_path``          **required in practice**; a local umt5-xxl directory. The
                            checkpoint's own ``config.json`` points at a training
                            machine and RLinf's model YAML ships ``null``.
``dreamzero_root``          the ``dreamzero`` checkout that provides ``groot``; it is
                            normally already on ``PYTHONPATH``.
``embodiment_tag``          default ``libero_sim``.
``action_horizon``          actions per chunk; must be >= the protocol's replan_steps.
``target_video_height/width`` default taken from the checkpoint's ``config.json``.
``rng_mode``                always ``fixed_per_predict``: ``WANPolicyHead`` draws its
                            noise from a fixed config seed (1140), so repeats of one
                            initial state are not independent samples.
``hash_checkpoint``         default ``True``; set ``false`` to skip hashing the shards.

Sparse-acceleration case study (CVPR'27 Sparse-rethinking) options:

``casestudy`` / ``casestudy_file``  the arm: ``{method: dense|anchor, keep_ratio, selection:
                            action|random, dense_first_frames, recent_dense_frames,
                            num_dit_steps, seed_offset, seed_mode}``. ``anchor`` uses the
                            dreamzero fork's action-conditioned anchor key routing on the
                            LIBERO 5x10 two-view grid (shared V->V / A->V key set, route
                            built on the first DiT step of a call and reused).
``shadow_casestudy``        second arm evaluated on the same observation and seed per
                            call; logged, never executed.
``episode_plans``           per-call arm overrides ({episode_id: {"calls": {...},
                            "from_call": {...}}}), as in the FastWAM adapter.
With ``seed_mode: episode_call_hash`` (default) the head's noise seed is set per call
from (instruction, episode without repeat, call index) instead of the fixed 1140.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from action_eval_sdk import (
    ACTION_PROFILE_LIBERO_OSC_POSE_V1,
    OBSERVATION_PROFILE_LIBERO_2CAM_V1,
    RNG_MODE_FIXED_PER_PREDICT,
    PolicyDescription,
    Prediction,
)

try:
    from action_eval_sdk import RNG_MODE_EPISODE_CALL_HASH
except ImportError:  # pragma: no cover
    RNG_MODE_EPISODE_CALL_HASH = "episode_call_hash"

# DiT step masks of WANPolicyHead.__init__ (16 scheduler steps), by NUM_DIT_STEPS.
_DIT_MASKS = {
    5: [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False],
    6: [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True],
    7: [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True],
    8: [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True],
    16: [True] * 16,
}
_CS_DEFAULTS = {
    "method": "dense", "keep_ratio": 0.5, "selection": "action", "dense_first_frames": 1,
    "recent_dense_frames": 0, "num_dit_steps": 8, "seed_offset": 0, "seed_mode": "episode_call_hash",
}


def _parse_casestudy(raw) -> dict:
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(_CS_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown casestudy keys: {unknown}")
    cs = {**_CS_DEFAULTS, **raw}
    if cs["method"] not in ("dense", "anchor"):
        raise ValueError("casestudy.method must be dense or anchor")
    if cs["selection"] not in ("action", "random"):
        raise ValueError("casestudy.selection must be action or random")
    if int(cs["num_dit_steps"]) not in _DIT_MASKS:
        raise ValueError(f"num_dit_steps must be one of {sorted(_DIT_MASKS)}")
    if not 0.0 < float(cs["keep_ratio"]) <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    return cs


def _call_seed(episode_key: str, call_index: int, offset: int = 0) -> int:
    """Same per-call seed function as the FastWAM case-study adapter."""
    digest = hashlib.sha256(f"{episode_key}|{int(call_index)}|{int(offset)}".encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def _cs_hash(cs: dict) -> str:
    return hashlib.sha256(json.dumps(cs, sort_keys=True).encode()).hexdigest()[:16]

#: Populated on first use so importing this file for a contract check needs no torch.
_RLINF: dict[str, object] = {}

_DIGEST_CACHE: dict[tuple[str, int, int], str | None] = {}


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _cached_sha256(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGEST_CACHE:
        _DIGEST_CACHE[key] = _sha256(path)
    return _DIGEST_CACHE[key]


def checkpoint_digest(checkpoint: Path) -> dict[str, str]:
    """Digest every shard plus the config, so the run names exactly what loaded."""
    digests: dict[str, str] = {}
    for name in sorted(p.name for p in checkpoint.glob("*.safetensors")):
        value = _cached_sha256(checkpoint / name)
        if value:
            digests[name] = value
    for name in ("config.json", "model.safetensors.index.json"):
        target = checkpoint / name
        if target.is_file():
            value = _cached_sha256(target)
            if value:
                digests[name] = value
    return digests


def _promote_scalars_before_load() -> None:
    """Promote 0-d parameters at construction, not after the state dict is loaded.

    ``rlinf.models.embodiment.dreamzero.get_model`` builds the model, calls
    ``load_state_dict(strict=False)`` and only then calls
    ``_promote_scalar_params_to_1d``. The published checkpoint is the opposite order:
    it was converted from an FSDP model whose scalars had already been promoted, so
    ``action_head.image_encoder.model.log_scale`` is stored with shape ``[1]`` while a
    freshly built model has it as a 0-d parameter. ``strict=False`` forgives missing
    and unexpected keys but not a size mismatch, so the stock order fails on exactly
    that tensor.

    Promoting inside the constructor puts the shapes right *before* the load, and
    makes ``get_model``'s later promotion a no-op. This is the whole workaround: the
    model is still built and loaded by RLinf's own code.
    """
    from rlinf.models.embodiment.dreamzero import _promote_scalar_params_to_1d
    from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy

    if getattr(DreamZeroPolicy, "_promote_before_load", False):
        return
    original_init = DreamZeroPolicy.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _promote_scalar_params_to_1d(self)

    DreamZeroPolicy.__init__ = __init__
    DreamZeroPolicy._promote_before_load = True


def _rl_inf(model_root: Path):
    """Import RLinf's DreamZero builder from the declared checkout."""
    if _RLINF:
        return _RLINF
    import sys

    root = str(model_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from rlinf.models.embodiment.dreamzero import get_model
    from rlinf.models.embodiment.dreamzero.dreamzero_config import (
        validate_dreamzero_sft_model_cfg,
    )

    _promote_scalars_before_load()
    _RLINF["get_model"] = get_model
    _RLINF["validate"] = validate_dreamzero_sft_model_cfg
    return _RLINF


class DreamZeroPolicy:
    """RLinf DreamZero policy behind the action-eval contract."""

    def __init__(self, config: dict, device: str):
        self.config = dict(config)
        self.device = device
        self.model_root = Path(self.config.get("model_root") or os.getcwd()).expanduser()

        model_config = self.config.get("model_config")
        if not model_config:
            raise ValueError(
                "DreamZero adapter requires `model_config`: the RLinf model YAML, "
                "e.g. examples/embodiment/config/model/dreamzero_5b.yaml"
            )
        model_config = Path(model_config).expanduser()
        if not model_config.is_absolute():
            model_config = self.model_root / model_config
        if not model_config.is_file():
            raise FileNotFoundError(f"RLinf model config not found: {model_config}")
        self.model_config_path = model_config

        checkpoint = self.config.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "DreamZero adapter requires `checkpoint`: the checkpoint directory "
                "containing config.json and model.safetensors.index.json"
            )
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = self.model_root / checkpoint
        index = checkpoint / "model.safetensors.index.json"
        if not index.is_file():
            # Without the index RLinf silently builds a component-initialised model
            # instead of loading the fine-tuned weights. That would still produce a
            # success rate, so it must be a hard error here.
            raise FileNotFoundError(
                f"missing {index}; refusing to build a model that would not be the "
                "fine-tuned checkpoint"
            )
        self.checkpoint_path = checkpoint

        checkpoint_config = json.loads((checkpoint / "config.json").read_text())
        options = self.config

        tokenizer_path = options.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError(
                "DreamZero adapter requires `tokenizer_path` to a local umt5-xxl "
                "directory: the checkpoint's own path points at its training machine "
                "and the RLinf YAML ships null."
            )
        self.tokenizer_path = str(Path(tokenizer_path).expanduser())
        if not Path(self.tokenizer_path).is_dir():
            raise FileNotFoundError(f"tokenizer_path is not a directory: {self.tokenizer_path}")

        dreamzero_root = options.get("dreamzero_root")
        if dreamzero_root:
            import sys

            root = str(Path(dreamzero_root).expanduser())
            if root not in sys.path:
                sys.path.insert(0, root)

        import torch  # noqa: PLC0415 - keeps the module importable without torch
        from omegaconf import OmegaConf  # noqa: PLC0415

        self._torch = torch
        if torch.cuda.is_available() is False:
            raise RuntimeError(
                "DreamZero adapter requires CUDA: the patched WAN VAE places its "
                "normalisation statistics on 'cuda' during construction, so a CPU "
                "run cannot even build the model."
            )

        architecture = OmegaConf.load(model_config)
        # The YAML's interpolations are relative (``${....key}``), so it has to sit at
        # the same depth the RLinf eval config gives it: cfg.rollout.model.
        cfg = OmegaConf.create({"rollout": {"model": architecture}})
        overrides = {
            "model_type": "dreamzero",
            "precision": "bf16",
            "is_lora": False,
            "model_path": str(checkpoint),
            "metadata_json_path": str(checkpoint / "experiment_cfg" / "metadata.json"),
            "tokenizer_path": self.tokenizer_path,
            "embodiment_tag": str(options.get("embodiment_tag", "libero_sim")),
            # These four come from the checkpoint's own config.json: RLinf's model YAML
            # ships different defaults (24/24/176) which would change the token grid.
            "action_horizon": int(options.get("action_horizon", checkpoint_config["action_horizon"])),
            "num_action_per_block": int(
                options.get("num_action_per_block", checkpoint_config["num_action_per_block"])
            ),
            "num_action_chunks": int(
                options.get("num_action_chunks", checkpoint_config["action_horizon"])
            ),
            "target_video_height": int(
                options.get("target_video_height", checkpoint_config["target_video_height"])
            ),
            "target_video_width": int(
                options.get("target_video_width", checkpoint_config["target_video_width"])
            ),
        }
        for key, value in overrides.items():
            OmegaConf.update(cfg, f"rollout.model.{key}", value, merge=False)

        self._validate = _rl_inf(self.model_root)["validate"]
        model_cfg = self._validate(cfg.rollout.model)

        self._evaluation = {
            "action_horizon": int(model_cfg.get("action_horizon")),
            "num_action_per_block": int(model_cfg.get("num_action_per_block")),
            "target_video_height": int(model_cfg.get("target_video_height")),
            "target_video_width": int(model_cfg.get("target_video_width")),
            "num_inference_steps": None,
        }

        build = _rl_inf(self.model_root)["get_model"]
        self.policy = build(model_cfg, torch.bfloat16)
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        head = getattr(self.policy, "action_head", None)
        if head is not None and hasattr(head, "num_inference_steps"):
            self._evaluation["num_inference_steps"] = int(head.num_inference_steps)
            self._evaluation["cfg_scale"] = float(getattr(head, "cfg_scale", 0.0))
            self._evaluation["seed"] = int(getattr(head, "seed", -1))

        # Compile and capture CUDA graphs *before* the benchmark creates its first
        # renderer context. The first real predict otherwise pays ~150 s of Dynamo
        # compilation on the same device while robosuite is creating and using an EGL
        # context, and on this host that overlap coincides with the native
        # ``mjr_readPixels`` abort: a DreamZero lane restarted every nine minutes
        # without completing a single episode, while the same model ran a whole
        # episode when it had the card to itself.
        if bool(options.get("warmup", True)):
            self._warmup()

        self._metadata_path = Path(model_cfg.get("metadata_json_path"))
        self._action_horizon = self._evaluation["action_horizon"]

        # -- case study --
        raw_arm, self.arm_name = options.get("casestudy"), None
        if options.get("casestudy_file"):
            import yaml  # noqa: PLC0415

            arm = yaml.safe_load(Path(options["casestudy_file"]).read_text()) or {}
            self.arm_name, raw_arm = arm.get("name"), arm.get("casestudy")
        self.casestudy = _parse_casestudy(raw_arm)
        self.shadow_cs = (
            None if options.get("shadow_casestudy") is None else _parse_casestudy(options["shadow_casestudy"])
        )
        self.shadow_executed = int(options.get("shadow_executed", 10))
        # Exact execution options (off by default): cross_kv_cache = per-call cross-attention K/V cache
        # in the DreamZero DiT (dreamzero branch perf); must reproduce the default path bit for bit.
        self.execution = dict(options.get("execution") or {})
        unknown = set(self.execution) - {"cross_kv_cache"}
        if unknown:
            raise ValueError(f"unknown execution options: {sorted(unknown)}")
        if self.execution.get("cross_kv_cache"):
            self.policy.action_head.exact_cross_kv_cache = True
        self.episode_plans = {}
        if options.get("episode_plans"):
            self.episode_plans = json.loads(Path(options["episode_plans"]).read_text())
        self._plan: dict = {}
        self._episode_key = ""
        self._dit = getattr(self.policy.action_head, "model", None)
        self._rng_mode = (
            RNG_MODE_EPISODE_CALL_HASH if self.casestudy["seed_mode"] == "episode_call_hash"
            else RNG_MODE_FIXED_PER_PREDICT
        )
        self._hash_checkpoint = bool(options.get("hash_checkpoint", True))
        self.predict_calls = 0
        self._fingerprint: dict[str, object] = {}

    # -- contract ----------------------------------------------------------
    def _warmup(self) -> None:
        """Run one throwaway inference so compilation is not concurrent with rendering.

        The dummy observation uses the benchmark's own shapes (256x256 uint8 cameras,
        an 8-dim state, a short instruction) so that Dynamo compiles the same graph the
        real calls will use. Nothing about the returned actions is kept.
        """
        import numpy as np  # noqa: PLC0415

        images = {
            "agentview": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
        }
        env_obs = {
            "main_images": images["agentview"][None],
            "wrist_images": images["wrist"][None],
            "states": np.zeros((1, 8), dtype=np.float32),
            "task_descriptions": ["warm up the policy"],
        }
        try:
            self.policy.predict_action_batch(env_obs, mode="eval")
        except Exception as exc:  # a failed warm-up must not look like a bad model
            print(f"[dreamzero-adapter] warm-up failed: {type(exc).__name__}: {exc}", flush=True)

    def describe(self) -> PolicyDescription:
        fingerprint = {
            "adapter": "rlinf-dreamzero",
            "identity": (
                "RLinf DreamZero WAN2.2-5B LIBERO-SFT; NOT DreamZero-DROID 14B"
            ),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_index": _cached_sha256(
                self.checkpoint_path / "model.safetensors.index.json"
            ),
            "checkpoint_config": _cached_sha256(self.checkpoint_path / "config.json"),
            "checkpoint_shards": (
                checkpoint_digest(self.checkpoint_path) if self._hash_checkpoint else None
            ),
            "checkpoint_size_bytes": sum(
                p.stat().st_size for p in self.checkpoint_path.glob("*.safetensors")
            ),
            "metadata_json": str(self._metadata_path),
            "metadata_json_sha256": _cached_sha256(self._metadata_path),
            "model_config": str(self.model_config_path),
            "model_config_sha256": _cached_sha256(self.model_config_path),
            "tokenizer_path": self.tokenizer_path,
            "embodiment_tag": "libero_sim",
            "action_horizon": self._action_horizon,
            "target_video": [
                self._evaluation["target_video_height"],
                self._evaluation["target_video_width"],
            ],
            "num_inference_steps": self._evaluation["num_inference_steps"],
            "cfg_scale": self._evaluation.get("cfg_scale"),
            "model_seed": self._evaluation.get("seed"),
            "rng_mode": self._rng_mode,
            "arm": self.arm_name,
            "casestudy": self.casestudy,
            "casestudy_hash": _cs_hash(self.casestudy),
            "shadow": self.shadow_cs,
            "execution": self.execution,
            "episode_plans_count": len(self.episode_plans),
        }
        self._fingerprint = fingerprint
        return PolicyDescription(
            observation_profile=OBSERVATION_PROFILE_LIBERO_2CAM_V1,
            action_profile=ACTION_PROFILE_LIBERO_OSC_POSE_V1,
            rng_mode=self._rng_mode,
            action_horizon=self._action_horizon,
            fingerprint=fingerprint,
            notes=(
                "RLinf DreamZeroPolicy behind a thin adapter: the benchmark flips the "
                "images and RLinf's own transform chain crops, resizes, concatenates "
                "the views, wraps the prompt and applies q99 normalisation, so this "
                "adapter adds no preprocessing; the model seeds every sampling call "
                "from its config, so repeats of one initial state are not independent"
            ),
        )

    def reset(self, episode: dict) -> None:
        """Clear the per-episode caches the action head keeps on the object.

        The head re-encodes the image every call because the observation is a single
        frame, but it caches ``language``, ``clip_feas``, ``ys`` and
        ``current_start_frame`` across calls. Clearing them keeps one episode from
        inheriting the previous episode's conditioning.
        """
        head = getattr(self.policy, "action_head", None)
        if head is not None:
            for attribute in ("language", "clip_feas", "ys", "current_start_frame"):
                if hasattr(head, attribute):
                    setattr(head, attribute, None)
        self.predict_calls = 0
        import re  # noqa: PLC0415

        episode_id = str(episode.get("episode_id", ""))
        self._episode_key = f"{episode.get('instruction', '')}|{re.sub(r'-r[0-9]+', '', episode_id)}"
        self._plan = self.episode_plans.get(episode_id, {})

    # -- case study helpers --------------------------------------------------
    def _arm_for_call(self, call_index: int):
        calls = self._plan.get("calls") or {}
        if str(call_index) in calls:
            return _parse_casestudy(calls[str(call_index)]), f"call:{call_index}"
        frm = self._plan.get("from_call")
        if frm and call_index >= int(frm["call"]):
            return _parse_casestudy(frm["casestudy"]), f"from:{frm['call']}"
        return self.casestudy, None

    def _apply_arm(self, cs: dict, seed: int) -> None:
        head = self.policy.action_head
        head.seed = int(seed)
        head.dit_step_mask = list(_DIT_MASKS[int(cs["num_dit_steps"])])
        if self._dit is not None and hasattr(self._dit, "configure_anchor_sparse_attention"):
            if cs["method"] == "anchor":
                self._dit.configure_anchor_sparse_attention(
                    enabled=True, keep_ratio=float(cs["keep_ratio"]),
                    recent_dense_frames=int(cs["recent_dense_frames"]),
                    view_layout="libero_side_by_side", grid_height=5, grid_width=10,
                    dense_first_frames=int(cs["dense_first_frames"]), selection=cs["selection"],
                    random_seed=(int(seed) + 1) & 0x7FFFFFFF, record_diagnostics=True, reuse_denoise=True,
                )
            else:
                self._dit.configure_anchor_sparse_attention(enabled=False)
            # The route cache key repeats across calls (current_start_frame resets to 0).
            self._dit.clear_anchor_sparse_route_cache()
        elif cs["method"] == "anchor":
            raise RuntimeError("this dreamzero checkout has no anchor sparse attention")

    def _infer(self, env_obs: dict, cs: dict, seed: int):
        self._apply_arm(cs, seed)
        actions, _result = self.policy.predict_action_batch(env_obs, mode="eval")
        diag: dict = {"seed": int(seed), "method": cs["method"], "num_dit_steps": int(cs["num_dit_steps"])}
        if cs["method"] == "anchor" and self._dit is not None:
            route = self._dit.get_last_anchor_route()
            if route is not None and getattr(route, "scores", None) is not None:
                import torch  # noqa: PLC0415

                scores = route.scores.float()  # [B, F, 50]
                first = int(cs["dense_first_frames"])
                frame_seq = scores.shape[-1]
                idx = route.video_indices[0].long()
                sparse = idx[(idx >= first * frame_seq)]
                flat = scores[0].reshape(-1)
                total = flat[first * frame_seq:].clamp_min(0).sum()
                if float(total) > 0:
                    diag["kept_score_mass_fraction"] = float(flat[sparse].clamp_min(0).sum() / total)
                diag["kept_video_keys"] = int(idx.numel())
        return np.asarray(actions, dtype=np.float32), diag

    def predict(self, observation: dict) -> Prediction:
        import time  # noqa: PLC0415

        images = observation.get("images") or {}
        state = np.asarray(observation.get("state"))
        instruction = observation.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("observation is missing a task instruction")
        if state.shape != (8,):
            raise ValueError(f"expected an 8-dim proprio vector, got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("proprio vector contains non-finite values")
        expected = {"agentview", "wrist"}
        if set(images) != expected:
            raise ValueError(f"expected cameras {sorted(expected)}, got {sorted(images)}")
        for name, image in images.items():
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"camera {name} is {image.dtype} with shape {image.shape}; the "
                    "observation profile guarantees uint8 [H,W,3]"
                )

        env_obs = {
            # Order is part of the trained layout: exterior left, wrist right.
            "main_images": np.ascontiguousarray(images["agentview"])[None],
            "wrist_images": np.ascontiguousarray(images["wrist"])[None],
            "states": state.astype(np.float32)[None],
            "task_descriptions": [instruction],
        }

        call_index = self.predict_calls
        cs, override = self._arm_for_call(call_index)
        if cs["seed_mode"] == "episode_call_hash":
            seed = _call_seed(self._episode_key, call_index, int(cs["seed_offset"]))
        else:
            seed = 1140
        started = time.monotonic()
        actions, diag = self._infer(env_obs, cs, seed)
        elapsed = time.monotonic() - started
        if override is not None:
            diag["plan_override"] = {"label": override, "casestudy_hash": _cs_hash(cs)}
        if self.shadow_cs is not None:
            # The shadow draws its own noise (its seed_offset), as in the FastWAM adapter.
            shadow_seed = (_call_seed(self._episode_key, call_index, int(self.shadow_cs["seed_offset"]))
                           if self.shadow_cs["seed_mode"] == "episode_call_hash" else seed)
            shadow, sdiag = self._infer(env_obs, self.shadow_cs, shadow_seed)
            a, b, k = actions[0], shadow[0], self.shadow_executed
            d = a - b
            diag["shadow"] = {
                "rms_executed": float(np.sqrt(np.mean(d[:k, :6] ** 2))),
                "rms_full": float(np.sqrt(np.mean(d[:, :6] ** 2))),
                "max_abs_executed": float(np.abs(d[:k, :6]).max()),
                "gripper_disagree_executed": int(np.sum(np.sign(a[:k, -1]) != np.sign(b[:k, -1]))),
                "translation_diff_executed": float(np.linalg.norm(a[:k, :3].sum(0) - b[:k, :3].sum(0))),
                "rotation_diff_executed": float(np.linalg.norm(a[:k, 3:6].sum(0) - b[:k, 3:6].sum(0))),
            }

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(f"expected a [1,T,7] action batch, got {actions.shape}")
        chunk = actions[0]
        if chunk.shape[1] != 7:
            raise ValueError(f"expected 7 action dimensions, got {chunk.shape[1]}")
        if not np.isfinite(chunk).all():
            raise ValueError("DreamZero produced non-finite actions")
        self.predict_calls += 1
        return Prediction(
            actions=chunk,
            timing={"predict_seconds": round(elapsed, 6)},
            diagnostics={
                "horizon": int(chunk.shape[0]),
                "predict_call_index": self.predict_calls,
                "call_index": call_index,
                **diag,
            },
        )

    def close(self) -> None:
        policy = getattr(self, "policy", None)
        if policy is not None and hasattr(policy, "action_head"):
            setattr(policy, "action_head", None)
        self.policy = None
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_policy(config: dict, device: str):
    return DreamZeroPolicy(config, device)
