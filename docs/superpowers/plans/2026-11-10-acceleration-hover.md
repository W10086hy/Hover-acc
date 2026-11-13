# Acceleration Hover Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution and verification.

**Goal** Achieve stable 120 s hover at the per-case position limits in the approved acceleration route.

**Architecture** Reuse P2's flexible-plant trajectory, adjoint and checkpoint loader. Add an opt-in continuation protocol with independent seeds and an audited native comparison.

**Tech stack** PyTorch CUDA, MuJoCo, Python unittest.

**Spec** [Hover design](../specs/2026-11-10-acceleration-hover-design.md)

## Global constraints

- Preserve the parent Actor and Adam and keep the acceleration interface and onboard inputs.
- Keep training at 30 s with 10 s credit and 1 s recomputation. Apply the 120 s gates to frozen native runs.
- Score safety, three-dimensional RMSE, and the complete last 30 s for every case.

## Tasks

- [x] Add a failing test for per-case gating, including an unsafe case, a weak failure hidden by the average, and a strong case that passes RMSE but drifts. Implement the gate and rerun the test.
- [x] Add a continuation protocol and restore the exact parent Actor and Adam with parameter-order verification. Run a short finite-gradient preflight and record its measured runtime.
- [x] Reuse the native evaluator with declared steady winds. Compare the parent and accepted candidates, revise only using development data, then freeze before the separate validation suite.
- [x] Recompute all 100 Hz metrics and inspect complete 3D and XY replays. Record the checkpoint, scope, and any unmet criteria in the project report.

The initial selection baseline used all 16 paired cases for 120 s. All completed safely and met the tail motion criteria. The strongest weak-wind error was 0.8634 m, with 4 of 8 weak-wind cases above 0.50 m. The 1 s preflight took 9.53 s on the RTX 3060. The first 30 s gradient took 298.03 s and the accepted update reduced the maximum 30 s selection error in weak wind from 0.5620 to 0.5095 m.

The first continuation checkpoint reached 6 of 8 weak-wind passes over 120 s. All 16 cases remained safe and met the tail criteria, but the weakest case remained at 0.8372 m. A second short-horizon update reduced its selection maximum to 0.4676 m, while the long-horizon offset remained the development focus.

A paired 60 s probe with the first continuation checkpoint gave a weak-wind maximum of 0.4649 m with stationary accelerometer offset calibration, 0.6582 m with a small horizontal output bias initialization alone, and 0.3687 m with both. The calibrated continuation therefore estimates a body accelerometer offset from 2,000 stationary onboard samples and retains the existing measurement noise. The horizontal head offset is recorded as initialization. The 30 s BPTT window begins after a 60 s causal prefix. The chosen configuration is qualified, trained and assessed through the same accelerometer correction in the differentiable and native simulation paths. Hardware must supply stationary preflight accelerometer and attitude samples.
