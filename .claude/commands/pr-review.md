---
name: pr-review
description: PR code review with risk-based analysis for CV / calibration / data-collection code
allowed-tools: Read, Grep, Glob, Bash, Task
---

# PR Code Review

Code review for the current branch's Pull Request, tailored to this project's
computer-vision, calibration, and data-collection code.

## Arguments

`$ARGUMENTS`

- No arguments: Review PR for current branch
- PR number: Review specific PR (e.g., `/pr-review 123`)
- `--quick`: Quick mode, only run summary analysis

## Quick Start

1. Get current branch PR: `gh pr view --json number,title,state,isDraft`
1. If PR doesn't exist or is closed, stop and explain
1. Execute Phases 1-3 in order

## Workflow Overview

```
Phase 1: PR Analysis
    ├─ PR Status Check
    ├─ Change Summary
    └─ Risk Classification
    ↓
Phase 2: Targeted Review
    ├─ Numerics / Geometry Review (if calibration or pose math changed)
    ├─ Pipeline Review (if RigPipeline / runners changed)
    ├─ ROS2 Review (if ros2_ws driver changed)
    ├─ API Review (if public interfaces changed)
    └─ General Code Review
    ↓
Phase 3: Summary Report
```

______________________________________________________________________

## Phase 1: PR Analysis

### 1.1 Get PR Info

```bash
gh pr view --json number,title,body,files,additions,deletions
gh pr diff
```

### 1.2 Risk Classification

Classify each changed file by risk level:

| Risk Level | File Patterns                                                                  | Review Depth |
| ---------- | ------------------------------------------------------------------------------ | ------------ |
| CRITICAL   | `calibration/fisheye.py`, `calibration/pinhole.py`, `core/geometry.py`         | Full trace   |
| HIGH       | `pose_6d/estimator.py`, `pose_6d/known_board.py`, `core/markers.py`, `core/rectify.py` | Thorough |
| MEDIUM     | `gripper/`, `runners/`, `core/camera/`, `core/viz/`, `calibration/{auto,capture,two_stage}.py` | Standard |
| LOW        | `scripts/`, `config/`, `ros2_ws/` (build / launch tweaks), docs, `.claude/`    | Basic        |

### 1.3 Numerics & Frame Risks

Flag these patterns as CRITICAL or HIGH:

- Changes to camera intrinsics / distortion-model math
- Coordinate-frame conversions (world ↔ camera, fisheye ↔ pinhole, plane ↔ image)
- Unit changes (degrees ↔ radians, meters ↔ millimeters, pixels ↔ normalized)
- Marker detection thresholds or rejection criteria
- Axis or handedness conventions
- Anything that touches `npz` calibration on-disk formats (silent breaks)

______________________________________________________________________

## Phase 2: Targeted Review

### Numerics / Geometry Review (CRITICAL/HIGH risk files)

Check for:

- [ ] Units consistent end-to-end (no implicit deg↔rad or m↔mm conversions)
- [ ] Coordinate frames documented at function boundaries
- [ ] No silent NaN/Inf propagation (guard near-singular matrices, zero norms)
- [ ] Axis / handedness convention matches the rest of the codebase
- [ ] On-disk calibration formats (`.npz`) remain loadable, or migration is documented
- [ ] Numerical tolerances are explicit, not magic numbers
- [ ] Vectorized ops behave correctly on empty inputs

### Pipeline Review (RigPipeline / runners / replay changes)

Check for:

- [ ] Stage outputs match the contract expected by downstream stages
- [ ] Frame-dropping / missing-detection paths handled (return NaN, skip, etc. — not crash)
- [ ] Viser overlay state cleaned up between frames
- [ ] No accidental tight loops without yielding (replay runs at near real-time)
- [ ] Default test video (`data/aruco_test/VID_*.insv`) still replays end-to-end

### ROS2 Review (`ros2_ws/` changes — insta360 driver)

Check for:

- [ ] CMakeLists / package.xml changes match the dep being added
- [ ] No new hard-coded topic names; respect existing conventions
- [ ] QoS profiles explicitly set (don't rely on defaults for image topics)
- [ ] Clean shutdown on `KeyboardInterrupt` / `rclcpp::shutdown()`
- [ ] Driver still builds with `pixi run build`

### API Review (public interface changes)

Check for:

- [ ] Backward compatibility maintained (or documented as breaking)
- [ ] Type hints present on new / modified public functions
- [ ] Dataclass / config field changes mirrored anywhere they're loaded from YAML
- [ ] Imports updated at every call site (no stale paths)

### General Code Review

Check for:

- [ ] No wildcard imports
- [ ] No hardcoded paths, IPs, or device serials
- [ ] Proper exception handling (not swallowing exceptions)
- [ ] Resource cleanup (file handles, video readers, viser servers)
- [ ] Type annotations present
- [ ] Comments explain *why*, not *what* — no narration of obvious code

______________________________________________________________________

## Phase 3: Summary Report

```markdown
# PR Review Summary

## PR Overview
- **Title**: PR title
- **Risk Level**: CRITICAL | HIGH | MEDIUM | LOW
- **Files Changed**: N

## Findings

### CRITICAL Issues
1. **[Title]** - `file.py:123`
   - Problem: [description]
   - Fix: [suggestion]

### Suggestions
1. **[Title]** - `file.py:456`
   - [description]

### Looks Good
- [positive observations]

## Review Statistics
- Total issues: X (CRITICAL: X, HIGH: X, MEDIUM: X, LOW: X)
```

______________________________________________________________________

## Important Notes

- **Do NOT** automatically post comments to PR
- Must provide file path and line number when referencing issues
- Use `gh` to interact with GitHub, not web fetch
- Pay special attention to geometry / calibration math — silent numerical bugs
  are the highest-impact failure mode in this codebase
