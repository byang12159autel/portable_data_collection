---
name: gen-commit-msg
description: Generate intelligent commit messages based on staged changes. Invoke with /gen-commit-msg.
---

# Generate Commit Message

Generate a well-formatted commit message based on staged changes.

## Usage

```
/gen-commit-msg [--amend] [--scope <scope>]
```

**Arguments:**

- `--amend`: Amend the previous commit instead of creating new
- `--scope <scope>`: Force a specific scope (e.g., `calibration`, `pose_6d`)

## Workflow

### Step 1: Analyze Changes

```bash
# Check staged files
git diff --cached --name-only

# Check staged content
git diff --cached

# Check recent commit style
git log --oneline -5
```

### Step 2: Categorize Changes

| Type       | When to Use                     |
| ---------- | ------------------------------- |
| `feat`     | New feature or capability       |
| `fix`      | Bug fix                         |
| `docs`     | Documentation only              |
| `refactor` | Code change without feature/fix |
| `test`     | Adding or fixing tests          |
| `chore`    | Build, deps, config changes     |
| `perf`     | Performance improvement         |

### Step 3: Determine Scope

Infer scope from changed files:

- `calibration/` → `calibration`
- `pose_6d/` → `pose_6d`
- `gripper/` → `gripper`
- `core/` → `core`
- `runners/` → `runners`
- `ros2_ws/` → `ros2`
- `scripts/` → `scripts`
- `config/` → `config`
- `.claude/` → `claude`
- `pixi.toml` → `deps`
- Multiple areas → omit scope or use broader term

### Step 4: Generate Message

**Format:**

```
<type>(<scope>): <subject>

<body>

[Optional sections:]
Key changes:
- change 1
- change 2

Refs: #123, #456
```

**Rules:**

- Subject: imperative mood, ~50-72 chars, no period
- Body: explain "why" not "what", wrap at 72 chars
- Key changes: bullet list of main modifications (for complex commits)
- Refs: reference issues/PRs if applicable
- **Run command**: if the commit changes runnable behavior (pipeline, runner,
  calibration script, ROS2 launch), append a fenced code block with the exact
  command used to verify it. Use the default test video / calibration paths
  from memory where applicable. Example:

  ````
  Verified with:

  ```
  pixi run python -m runners.rig_replay \
      --video data/aruco_test/VID_20260518_093406_00_010.insv \
      --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \
      --world-marker-configs config/aruco_bench.yaml \
      --hinge-marker-config config/chopsticks-v1.yaml
  ```
  ````

### Step 5: Split or Single Commit

Evaluate whether changes should be **one commit or multiple**:

- **Split** when staged changes touch unrelated concerns (e.g., a bug fix + a new feature, or deps + docs + code)
- **Keep as one** when all changes serve a single logical purpose
- Each commit should be atomic: it builds, passes lint, and makes sense on its own

If splitting, plan the commits in order and show all previews together. Use `git add -p` or specific file paths to stage each commit separately.

### Step 6: Confirm and Commit

Show preview (one block per commit if splitting):

```
─────────────────────────────────────
[1/2] chore(deps): pin pupil-apriltags to 1.0.4
...
─────────────────────────────────────
[2/2] feat(pose_6d): add learned-layout fallback for AprilTag boards
...
─────────────────────────────────────
```

Ask user to confirm, then execute each commit:

```bash
git add <files-for-commit-1>
git commit -m "$(cat <<'EOF'
<message>
EOF
)"

git add <files-for-commit-2>
git commit -m "$(cat <<'EOF'
<message>
EOF
)"
```

## Examples

**Single file fix:**

```
fix(calibration): guard against zero-determinant homographies

Skip frames whose homography decomposition produces a
near-singular rotation matrix instead of propagating NaNs
into the optimizer.
```

**Multi-file feature:**

```
feat(gripper): add hinge-angle estimation from dot pairs

Compute hinge angle from two tracked dots on the jaw using
the plane-space helpers in core/geometry. Wires into the
existing replay pipeline via HingeAngleEstimator.

Key changes:
- Add HingeAngleEstimator in gripper/estimator.py
- Add dot-pair helpers in gripper/dots.py
- Expose hinge overlay in runners/rig_replay.py
```

**Config change:**

```
chore(deps): pin pupil-apriltags to 1.0.4

Pin pupil-apriltags so detection thresholds stay
reproducible across machines.
```
