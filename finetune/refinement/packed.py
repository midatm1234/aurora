"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Shared construction helpers for the packed (unified) Phase-2 refiners.

``diffusion_unet``, ``diffusion_transformer`` and ``flow_matching_transformer``
all consume the same packed representation

    residual state   [N, C_res,  H, W]   (normalized target space)
    conditioning     [N, C_cond, H, W]   (packed rollout / statics / masks)
    process time     [N]                 (diffusion timestep or flow time)
    forecast lead    [N]                 (physical hours, separate embedding)

where ``N`` is ``batch * rollout_lead_times`` because the caller folds the
lead-time axis into the effective batch dimension. Everything specific to a
generative process lives in :mod:`finetune.refinement.diffusion` or
:mod:`finetune.refinement.flow_matching`.

Residual scaling
----------------
This class owns the boundary between the **normalized target space** (where the
residual is defined and where every caller and every loss operates) and the
**generative space** (where the diffusion / flow process runs). Subclasses only
ever see the generative space, so none of them has to know that the scaling
exists:

``compute_training_loss``
    encodes the residual, calls ``_training_loss``, then decodes every estimate
    before the auxiliary losses run.
``sample_residual`` / ``deterministic_residual``
    call ``_sample`` / ``_deterministic`` and decode the result.

See :mod:`finetune.refinement.residual_scaling` for why this is required rather
than optional.
"""

from __future__ import annotations

import contextlib

import torch

from finetune.refinement.backbones import ConditionalResidualUNet, SpatialResidualTransformer
from finetune.refinement.base import RefinerOutput, ResidualRefiner, masked_loss
from finetune.refinement.config import ConfigValidationError, RefinementConfig
from finetune.refinement.losses import compute_auxiliary_losses
from finetune.refinement.packing import FieldPacking
from finetune.refinement.residual_scaling import ResidualScaler
from finetune.refinement.vertical import VerticalChannelEncoder

__all__ = ["PackedRefiner"]


class PackedRefiner(ResidualRefiner):
    """Base class for the refiners that operate on packed multi-channel fields."""

    def __init__(
        self,
        config: RefinementConfig,
        *,
        residual_channels: int,
        cond_channels: int,
        metadata=None,
    ) -> None:
        super().__init__(
            config,
            residual_channels=residual_channels,
            cond_channels=cond_channels,
            metadata=metadata,
        )
        self.lead_time_conditioning = bool(config.conditioning.forecast_lead_time)
        self.lead_time_scale_hours = float(getattr(metadata, "lead_time_scale_hours", 72.0) or 72.0)
        self.lon_periodic = bool(getattr(metadata, "lon_periodic", False))
        # Calendar / geometry scalars fused into the conditioning vector, and
        # causal temporal-context channels appended to the spatial conditioning.
        # Both widths are part of the checkpoint contract: a network trained
        # with them cannot be evaluated without them, which `_metadata_for` and
        # `augment_conditioning` enforce at call time.
        self.calendar_features = int(config.conditioning.metadata_feature_count)
        self.temporal_context_channels = int(config.temporal.context_channels_if_active)
        self._frame_metadata: torch.Tensor | None = None
        self._frame_temporal_context: torch.Tensor | None = None

        # Continuous pressure / variable identity for the packed channel axis.
        # Lives here rather than on the two-phase wrapper because it modulates
        # the packed state and therefore belongs in the refiner's state dict.
        self.vertical_encoder: VerticalChannelEncoder | None = None
        if config.conditioning.vertical_identity:
            if not isinstance(metadata, FieldPacking):
                raise ConfigValidationError(
                    "refinement.conditioning.vertical_identity needs the FieldPacking "
                    "metadata to read each channel's pressure level, units and variable "
                    f"identity; got {type(metadata).__name__}."
                )
            self.vertical_encoder = VerticalChannelEncoder(
                metadata, max(8, int(metadata.num_channels))
            )
        self.vertical_features = (
            0 if self.vertical_encoder is None else self.vertical_encoder.dim
        )
        #: Total width of the conditioning vector consumed by the backbone MLP.
        self.metadata_features = self.calendar_features + self.vertical_features
        target_space = config.target_space
        self.residual_scaler = ResidualScaler(
            self.residual_channels,
            mode=target_space.resolved_residual_scaling(),
            center=target_space.residual_scaling_center,
            momentum=target_space.residual_scaling_momentum,
            warmup_batches=target_space.residual_scaling_warmup_batches,
            target_std=target_space.residual_scaling_target_std,
        )
        self.residual_clip_standard_deviations = float(
            target_space.residual_clip_standard_deviations
        )
        self.net = self._build_net(config)
        # Version 1 used an independent full-size network for the deployed
        # conditional mean and let ``net`` model only stochastic innovations.
        # Version 2 shares the process network with the point product, so the
        # diffusion/flow objective and its deterministic query train the model
        # that is actually scored. Keep v1 as the omitted-key default so old
        # checkpoints remain byte-for-byte compatible.
        if config.deterministic_head == "separate_mean":
            self.mean_net: torch.nn.Module | None = self._build_net(config)
            self._zero_output_projection(self.mean_net)
            parameterization_version = 1
        else:
            self.mean_net = None
            parameterization_version = 2
        self.register_buffer(
            "innovation_parameterization_version",
            torch.tensor(parameterization_version, dtype=torch.long),
        )

    # -- construction ----------------------------------------------------
    def _build_net(self, config: RefinementConfig) -> torch.nn.Module:
        # The temporal context is appended to the spatial conditioning, so the
        # network's declared conditioning width includes it.
        cond_channels = self.cond_channels + self.temporal_context_channels
        if config.uses_transformer:
            t = config.transformer
            return SpatialResidualTransformer(
                in_channels=self.residual_channels,
                cond_channels=cond_channels,
                out_channels=self.residual_channels,
                patch_size=t.patch_size,
                embedding_dim=t.embedding_dim,
                num_heads=t.num_heads,
                num_blocks=t.num_blocks,
                mlp_ratio=t.mlp_ratio,
                dropout=t.dropout,
                positional_encoding=t.positional_encoding,
                max_tokens_lat=t.max_tokens_lat,
                max_tokens_lon=t.max_tokens_lon,
                attention_mode=t.attention_mode,
                window_size=t.window_size,
                shifted_windows=t.shifted_windows,
                gradient_checkpointing=t.gradient_checkpointing,
                optimized_attention=t.optimized_attention,
                zero_init_output=t.zero_init_output,
                lead_time_conditioning=self.lead_time_conditioning,
                lead_time_scale_hours=self.lead_time_scale_hours,
                lon_periodic=self.lon_periodic,
                local_refinement=t.local_refinement,
                metadata_features=self.metadata_features,
            )
        u = config.unet
        return ConditionalResidualUNet(
            in_channels=self.residual_channels,
            cond_channels=cond_channels,
            out_channels=self.residual_channels,
            hidden_channels=u.hidden_channels,
            num_levels=u.num_levels,
            num_residual_blocks=u.num_residual_blocks,
            time_embedding_dim=u.time_embedding_dim,
            dropout=u.dropout,
            bottleneck_attention=u.bottleneck_attention,
            attention_heads=u.attention_heads,
            zero_init_output=u.zero_init_output,
            lead_time_conditioning=self.lead_time_conditioning,
            lead_time_scale_hours=self.lead_time_scale_hours,
            lon_periodic=self.lon_periodic,
            metadata_features=self.metadata_features,
        )

    @staticmethod
    @torch.no_grad()
    def _zero_output_projection(network: torch.nn.Module) -> None:
        projection = getattr(network, "out_proj", None)
        if not isinstance(projection, torch.nn.Module):
            raise RuntimeError(
                f"{type(network).__name__} does not expose an out_proj for safe "
                "deterministic-mean initialization."
            )
        for parameter in projection.parameters(recurse=False):
            parameter.zero_()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Load old full-correction checkpoints without changing their outputs."""
        version_key = prefix + "innovation_parameterization_version"
        legacy_checkpoint = version_key not in state_dict
        if legacy_checkpoint:
            state_dict[version_key] = torch.zeros(
                (), dtype=self.innovation_parameterization_version.dtype
            )
        if self.mean_net is not None:
            mean_state = self.mean_net.state_dict()
            for name, value in mean_state.items():
                full_name = prefix + "mean_net." + name
                if full_name not in state_dict:
                    state_dict[full_name] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_attention_implementation(self, implementation: str) -> None:
        for network in (self.net, self.mean_net):
            if network is None:
                continue
            setter = getattr(network, "set_attention_implementation", None)
            if setter is None:
                raise AttributeError(
                    f"{type(network).__name__} has no configurable attention "
                    "implementation."
                )
            setter(implementation)

    # -- residual scaling ------------------------------------------------
    def fit_residual_scale(self, residual: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Initialise the residual statistics from a representative sample."""
        self.residual_scaler.fit(residual, mask)

    def freeze_residual_scale(self, value: bool = True) -> None:
        self.residual_scaler.freeze(value)

    def _guard_scaled_correction(self, scaled: torch.Tensor) -> torch.Tensor:
        """Clamp a complete correction in standardized training-residual units."""
        value = scaled.float()
        standard_deviations = self.residual_clip_standard_deviations
        if standard_deviations <= 0.0:
            return value
        if not self.residual_scaler.is_active:
            raise RuntimeError(
                "Residual amplitude clipping requires an active ResidualScaler."
            )
        # ResidualScaler maps the raw training standard deviation to target_std
        # in generative space, so k raw standard deviations equal
        # k * target_std scaled units.
        limit = standard_deviations * float(self.residual_scaler.target_std)
        return value.clamp(min=-limit, max=limit)

    def _decode_deployed_correction(self, scaled: torch.Tensor) -> torch.Tensor:
        """Apply the one scaled-space guard immediately before deployment decode."""
        return self.residual_scaler.decode(self._guard_scaled_correction(scaled))

    def guard_correction_normalized(self, correction: torch.Tensor) -> torch.Tensor:
        """Guard an already-decoded complete correction without changing units."""
        if self.residual_clip_standard_deviations <= 0.0:
            # Preserve the exact legacy arithmetic path when the new option is
            # omitted or explicitly disabled; even an encode/decode round trip
            # could otherwise introduce an avoidable rounding difference.
            return correction.float()
        scaled = self.residual_scaler.encode(correction)
        return self._decode_deployed_correction(scaled)

    # -- shared helpers --------------------------------------------------
    def _lead_for(self, forecast_lead_time: torch.Tensor | None) -> torch.Tensor | None:
        return forecast_lead_time if self.lead_time_conditioning else None

    @contextlib.contextmanager
    def use_frame_conditioning(
        self,
        *,
        metadata: torch.Tensor | None = None,
        temporal_context: torch.Tensor | None = None,
    ):
        """Bind per-frame calendar metadata and temporal context for this call.

        Both quantities describe the *forecast frame*, so they are constant
        across every evaluation of the generative process for that frame. They
        are bound once by the caller and consumed automatically by every
        internal ``net`` / ``mean_net`` call, which is what guarantees that a
        denoiser or velocity evaluation can never silently run without the
        conditioning its checkpoint was trained with.

        Binding rather than threading a parameter through
        ``_training_loss`` / ``_sample`` / ``_deterministic`` also keeps the
        subclass API unchanged for the diffusion and flow-matching processes.

        Neither argument may depend on the current noised residual: that is what
        makes the value reusable across solver evaluations. See
        :class:`finetune.refinement.temporal.TemporalContextCache`.
        """
        previous = (self._frame_metadata, self._frame_temporal_context)
        self._frame_metadata = metadata
        self._frame_temporal_context = temporal_context
        try:
            yield
        finally:
            self._frame_metadata, self._frame_temporal_context = previous

    def _expand_to_members(
        self, value: torch.Tensor, batch: int, *, name: str
    ) -> torch.Tensor:
        """Expand bound per-frame conditioning to a member-replicated batch.

        Batched ensemble generation replicates the conditioning with
        ``repeat_interleave``, giving the flat layout
        ``[b0m0, b0m1, ..., b0m(M-1), b1m0, ...]`` (see
        :meth:`AuroraTwoPhaseRefiner.refine`). Per-frame conditioning must
        follow *exactly* that layout: a plain ``repeat`` would pair frame 0's
        calendar with sample 1's field and silently corrupt the ensemble.
        """
        rows = int(value.shape[0])
        if rows == batch:
            return value
        if rows == 1:
            return value.expand(batch, *value.shape[1:])
        if batch % rows == 0:
            return value.repeat_interleave(batch // rows, dim=0)
        raise ValueError(
            f"Bound {name} has {rows} rows, which is neither the batch size "
            f"{batch} nor a divisor of it. Ensemble replication uses "
            "repeat_interleave, so the frame count must divide the packed batch."
        )

    def _metadata_for(self, batch: int) -> torch.Tensor | None:
        """Validated conditioning vector for a packed batch of ``batch`` rows.

        Concatenates the bound per-frame calendar scalars with the pooled
        vertical/variable-identity embedding. The vertical part is identical for
        every row (it describes the field set, not the frame) but is carried in
        the same vector so the fusion MLP can form calendar x level interactions.
        """
        if self.metadata_features <= 0:
            return None
        parts: list[torch.Tensor] = []
        if self.calendar_features > 0:
            metadata = self._frame_metadata
            if metadata is None:
                raise RuntimeError(
                    "This refiner was built with "
                    f"calendar_features={self.calendar_features} but no calendar "
                    "metadata is bound. Wrap the call in "
                    "`refiner.use_frame_conditioning(metadata=...)`; running without it "
                    "would train and deploy different conditioning contracts."
                )
            if metadata.shape[-1] != self.calendar_features:
                raise ValueError(
                    f"Bound calendar metadata must have {self.calendar_features} "
                    f"features, got {tuple(metadata.shape)}."
                )
            parts.append(self._expand_to_members(metadata, batch, name="calendar metadata"))
        if self.vertical_encoder is not None:
            pooled = self.vertical_encoder.pooled()
            parts.append(pooled[None].expand(batch, pooled.shape[0]))
        combined = torch.cat([p.to(parts[0].dtype) for p in parts], dim=-1)
        if combined.shape != (batch, self.metadata_features):
            raise RuntimeError(
                f"Conditioning vector must be [{batch}, {self.metadata_features}], "
                f"built {tuple(combined.shape)}."
            )
        return combined

    def augment_conditioning(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Append the bound temporal context to the spatial conditioning stack.

        The context is *extra conditioning channels*, so temporal information
        reaches the denoiser at every process evaluation rather than being
        applied to already-sampled frames.
        """
        context = self._frame_temporal_context
        if self.temporal_context_channels <= 0:
            if context is not None:
                raise RuntimeError(
                    "Temporal context was bound but this refiner was built with "
                    "temporal_context_channels=0. The conditioning width is part of "
                    "the checkpoint contract and cannot change at call time."
                )
            return conditioning
        if context is None:
            raise RuntimeError(
                "This refiner was built with temporal_context_channels="
                f"{self.temporal_context_channels} but no temporal context is bound. "
                "Wrap the call in `refiner.use_frame_conditioning(temporal_context=...)`."
            )
        batch = conditioning.shape[0]
        expected_tail = (self.temporal_context_channels, *conditioning.shape[-2:])
        if tuple(context.shape[1:]) != expected_tail:
            raise ValueError(
                f"Temporal context must have trailing shape {expected_tail}, got "
                f"{tuple(context.shape[1:])}."
            )
        context = self._expand_to_members(context, batch, name="temporal context")
        return torch.cat([conditioning, context.to(conditioning.dtype)], dim=1)

    def _evaluate_net(
        self,
        network: torch.nn.Module,
        state: torch.Tensor,
        conditioning: torch.Tensor,
        process_time: torch.Tensor,
        lead: torch.Tensor | None,
    ) -> torch.Tensor:
        """Single funnel for every generative-network evaluation.

        ``lead`` must already have passed through :meth:`_lead_for`.

        The per-channel vertical affine is applied to the network's *input*
        representation of the packed state. It is an input reparameterization
        only: the generative process still operates on the unmodified residual,
        and the affine is an exact identity at construction.
        """
        if self.vertical_encoder is not None:
            state = self.vertical_encoder.apply_channel_modulation(state)
        return network(
            state,
            self.augment_conditioning(conditioning),
            process_time,
            lead,
            self._metadata_for(int(conditioning.shape[0])),
        )

    @torch.no_grad()
    def _output_projection_is_exactly_zero(
        self, network: torch.nn.Module | None = None
    ) -> bool:
        """Whether a configured zero-init projection is still untouched."""
        selected = self.net if network is None else network
        projection = getattr(selected, "out_proj", None)
        if not isinstance(projection, torch.nn.Module):
            return False
        parameters = tuple(projection.parameters(recurse=False))
        if not parameters:
            return False
        return all(
            int(torch.count_nonzero(parameter).item()) == 0
            for parameter in parameters
        )

    def _deterministic_projection_is_exactly_zero(self) -> bool:
        """Whether deployed deterministic inference uses its identity shortcut."""
        if self.uses_innovation_parameterization:
            assert self.mean_net is not None
            return self._output_projection_is_exactly_zero(self.mean_net)
        if self.uses_shared_process_parameterization:
            return self._output_projection_is_exactly_zero(self.net)
        return False

    def _scaled_literal_zero(self, reference: torch.Tensor) -> torch.Tensor:
        """Represent a literal zero correction in centered generative space."""
        return self.residual_scaler.encode(torch.zeros_like(reference))

    def _mean_correction_scaled(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        differentiable: bool,
    ) -> torch.Tensor:
        """Direct supervised conditional mean in generative correction space."""
        if self.mean_net is None:
            raise RuntimeError(
                "The separate deterministic mean head is disabled; query the "
                "shared process network through _deterministic instead."
            )
        shape = self.residual_shape(conditioning)
        state = torch.zeros(
            shape, device=conditioning.device, dtype=conditioning.dtype
        )
        process_time = torch.zeros(
            shape[0], device=conditioning.device, dtype=torch.float32
        )
        context = torch.enable_grad() if differentiable else torch.no_grad()
        with context:
            prediction = self._evaluate_net(
                self.mean_net,
                state,
                conditioning,
                process_time,
                self._lead_for(forecast_lead_time),
            )
        if prediction.shape != state.shape:
            raise RuntimeError(
                "Deterministic mean head output shape must match correction shape; "
                f"got {tuple(prediction.shape)} and {tuple(state.shape)}."
            )
        prediction32 = prediction.float()
        if not bool(torch.isfinite(prediction32).all()):
            raise FloatingPointError(
                "Deterministic mean correction head produced non-finite values."
            )
        return prediction32

    @property
    def uses_innovation_parameterization(self) -> bool:
        return int(self.innovation_parameterization_version.item()) == 1

    @property
    def uses_shared_process_parameterization(self) -> bool:
        """Whether the deployed point correction comes from ``net`` itself."""
        return int(self.innovation_parameterization_version.item()) >= 2

    # -- public API (scaling boundary) -----------------------------------
    def compute_training_loss(
        self,
        residual_target: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        lead_index: torch.Tensor | None = None,
        area_weight: torch.Tensor | None = None,
        rollout_normalized: torch.Tensor | None = None,
    ) -> RefinerOutput:
        if conditioning.ndim != 4 or not conditioning.is_floating_point():
            raise ValueError(
                "Refiner conditioning must be a floating [N, C_cond, H, W] tensor; "
                f"got shape={tuple(conditioning.shape)}, dtype={conditioning.dtype}."
            )
        expected = self.residual_shape(conditioning)
        if tuple(residual_target.shape) != expected:
            raise ValueError(
                "Correction target shape must match packed conditioning batch/spatial "
                f"dimensions and residual channels; expected {expected}, got "
                f"{tuple(residual_target.shape)}."
            )
        if residual_target.device != conditioning.device:
            raise ValueError(
                "Correction target and conditioning must share a device; got "
                f"{residual_target.device} and {conditioning.device}."
            )
        if forecast_lead_time is not None and forecast_lead_time.numel() not in {
            1,
            conditioning.shape[0],
        }:
            raise ValueError(
                "forecast_lead_time must contain one value or one value per packed "
                f"sample; got {forecast_lead_time.numel()} for batch "
                f"{conditioning.shape[0]}."
            )

        # Observe only finite, masked training corrections. The scaler freezes
        # after its configured training calibration window.
        self.residual_scaler.observe(residual_target, mask)
        scaled_target = self.residual_scaler.encode(residual_target)

        loss_cfg = self.config.loss
        if self.uses_innovation_parameterization:
            mean_scaled = self._mean_correction_scaled(
                conditioning,
                forecast_lead_time=forecast_lead_time,
                differentiable=True,
            )
            # The stochastic process models only unpredictable departures from
            # the directly supervised conditional mean. Detaching the mean in
            # this target prevents the generative loss from moving the mean head
            # to make its own task artificially easier.
            process_target_scaled = scaled_target - mean_scaled.detach()
            mean_loss = masked_loss(
                mean_scaled, scaled_target, mask, kind="huber"
            )
        else:
            # A checkpoint without the version buffer predates the separate
            # mean/innovation contract and must retain full-correction semantics.
            mean_scaled = None
            mean_loss = None
            process_target_scaled = scaled_target

        output, process_estimate_scaled = self._training_loss(
            process_target_scaled,
            conditioning,
            forecast_lead_time=forecast_lead_time,
            mask=mask,
            generator=generator,
        )
        correction_estimate_scaled = process_estimate_scaled
        if mean_scaled is not None:
            correction_estimate_scaled = mean_scaled + process_estimate_scaled
        deployed_estimate_scaled = self._guard_scaled_correction(
            correction_estimate_scaled
        )
        correction_estimate = self.residual_scaler.decode(deployed_estimate_scaled)
        output.predicted_correction_normalized = correction_estimate
        output.predicted_residual = correction_estimate

        if not loss_cfg.has_auxiliary_terms:
            output.total_loss = output.generative_loss
        else:
            deterministic_estimate = mean_scaled
            deterministic_is_identity = False
            if deterministic_estimate is None and loss_cfg.needs_deterministic_estimate:
                deterministic_estimate = self._deterministic(
                    conditioning,
                    forecast_lead_time=forecast_lead_time,
                    differentiable=True,
                )
            if deterministic_estimate is not None:
                if self._deterministic_projection_is_exactly_zero():
                    # Inference returns a literal zero before centered decode.
                    # Use the same forward value for scientific auxiliaries but
                    # retain the network gradient via a straight-through shift.
                    identity = self._scaled_literal_zero(deterministic_estimate)
                    deterministic_estimate = deterministic_estimate + (
                        identity - deterministic_estimate
                    ).detach()
                    deterministic_is_identity = True
                if not deterministic_is_identity:
                    deterministic_estimate = self._guard_scaled_correction(
                        deterministic_estimate
                    )
            output = self._finish(
                output,
                residual_estimate=deployed_estimate_scaled,
                residual_target=scaled_target,
                mask=mask,
                lead_index=lead_index,
                area_weight=area_weight,
                rollout_normalized=rollout_normalized,
                deterministic_estimate=deterministic_estimate,
            )

        # A separate conditional mean is useful only when it is actually
        # supervised. New checkpoints therefore always receive a robust direct
        # mean objective. An explicit deterministic_weight controls its weight;
        # otherwise a safe weight of one closes the train/inference objective gap.
        if mean_loss is not None:
            generative = output.generative_loss
            total = output.total_loss
            assert generative is not None and total is not None
            configured_weight = float(loss_cfg.deterministic_weight)
            effective_weight = configured_weight if configured_weight > 0.0 else 1.0
            if configured_weight > 0.0 and output.deterministic_loss is not None:
                total = total - configured_weight * output.deterministic_loss
            total = total + effective_weight * mean_loss
            output.deterministic_loss = mean_loss
            output.total_loss = total
            output.diagnostics["deterministic"] = mean_loss.detach()
            output.diagnostics["mean_supervision_weight"] = torch.tensor(
                effective_weight, device=total.device, dtype=torch.float32
            )
            output.diagnostics["innovation_target_rms"] = (
                process_target_scaled.detach().float().square().mean().sqrt()
            )
        return output

    @torch.no_grad()
    def sample_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if self.uses_innovation_parameterization:
            mean = self.deterministic_residual(
                conditioning, forecast_lead_time=forecast_lead_time
            )
            innovation = self.sample_innovation_normalized(
                conditioning,
                forecast_lead_time=forecast_lead_time,
                generator=generator,
                num_steps=num_steps,
                mean_correction_normalized=mean,
            )
            return self.guard_correction_normalized(
                mean.float() + innovation.float()
            )
        if self._output_projection_is_exactly_zero(self.net):
            return torch.zeros(
                self.residual_shape(conditioning),
                device=conditioning.device,
                dtype=torch.float32,
            )
        scaled = self._sample(
            conditioning,
            forecast_lead_time=forecast_lead_time,
            generator=generator,
            num_steps=num_steps,
        )
        return self._decode_deployed_correction(scaled)

    @torch.no_grad()
    def sample_innovation_normalized(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
        mean_correction_normalized: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.uses_innovation_parameterization:
            return super().sample_innovation_normalized(
                conditioning,
                forecast_lead_time=forecast_lead_time,
                generator=generator,
                num_steps=num_steps,
                mean_correction_normalized=mean_correction_normalized,
            )
        if self._output_projection_is_exactly_zero(self.net):
            return torch.zeros(
                self.residual_shape(conditioning),
                device=conditioning.device,
                dtype=torch.float32,
            )
        scaled = self._sample(
            conditioning,
            forecast_lead_time=forecast_lead_time,
            generator=generator,
            num_steps=num_steps,
        )
        return self.residual_scaler.decode_difference(scaled)

    @torch.no_grad()
    def deterministic_residual(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._deterministic_projection_is_exactly_zero():
            # Scaled zero decodes to the fitted residual mean when centering is
            # active. An untouched zero-init head instead means "do no harm":
            # emit a literal zero correction, matching stochastic inference.
            return torch.zeros(
                self.residual_shape(conditioning),
                device=conditioning.device,
                dtype=torch.float32,
            )
        if self.uses_innovation_parameterization:
            assert self.mean_net is not None
            scaled = self._mean_correction_scaled(
                conditioning,
                forecast_lead_time=forecast_lead_time,
                differentiable=False,
            )
            return self._decode_deployed_correction(scaled)
        scaled = self._deterministic(
            conditioning, forecast_lead_time=forecast_lead_time, differentiable=False
        )
        return self._decode_deployed_correction(scaled)

    # -- subclass API (generative space) ---------------------------------
    def _training_loss(
        self,
        residual_target: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        mask: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[RefinerOutput, torch.Tensor]:
        """Return ``(output_with_generative_loss, clean_process_estimate)``.

        For current checkpoints, both values are stochastic innovations in the
        **scaled** generative space. Legacy checkpoints use complete corrections.
        """
        raise NotImplementedError

    def _sample(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        generator: torch.Generator | None,
        num_steps: int | None,
    ) -> torch.Tensor:
        """Draw one process sample in scaled generative space."""
        raise NotImplementedError

    def _deterministic(
        self,
        conditioning: torch.Tensor,
        *,
        forecast_lead_time: torch.Tensor | None,
        differentiable: bool = False,
    ) -> torch.Tensor:
        """Deterministic (mean-like) residual in the scaled generative space.

        ``differentiable=True`` is used by ``loss.deterministic_weight`` and must
        keep the autograd graph; the public wrapper runs under ``no_grad``.
        """
        raise NotImplementedError

    # -- auxiliary losses ------------------------------------------------
    def _finish(
        self,
        output: RefinerOutput,
        *,
        residual_estimate: torch.Tensor,
        residual_target: torch.Tensor,
        mask: torch.Tensor | None,
        lead_index: torch.Tensor | None,
        area_weight: torch.Tensor | None,
        rollout_normalized: torch.Tensor | None,
        deterministic_estimate: torch.Tensor | None = None,
    ) -> RefinerOutput:
        """Attach the enabled auxiliary terms and the total loss."""
        loss_cfg = self.config.loss
        terms = compute_auxiliary_losses(
            residual_estimate,
            residual_target,
            config=loss_cfg,
            mask=mask,
            packing=self.metadata if hasattr(self.metadata, "channels") else None,
            area_weight=area_weight,
            lead_index=lead_index,
            rollout_normalized=rollout_normalized,
            deterministic_estimate=deterministic_estimate,
            decode=self.residual_scaler.decode,
        )
        output.reconstruction_loss = terms.reconstruction
        output.bias_loss = terms.bias
        output.gradient_loss = terms.gradient
        output.pattern_correlation_loss = terms.pattern_correlation
        output.deterministic_loss = terms.deterministic
        output.mae_loss = terms.mae
        output.extreme_loss = terms.extreme
        output.peak_loss = terms.peak
        output.quantile_loss = terms.quantile
        output.variance_loss = terms.variance
        output.spectral_loss = terms.spectral
        output.degradation_loss = terms.degradation
        output.magnitude_loss = terms.magnitude
        output.diagnostics.update(terms.scalars())
        generative = output.generative_loss
        assert generative is not None
        total = generative
        if loss_cfg.has_auxiliary_terms:
            total = total + terms.weighted_total(loss_cfg, generative).to(generative.dtype)
        output.total_loss = total
        return output
