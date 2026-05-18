---
name: create-pr
description: Rebase from the latest `origin/main`, squash the commits from it, and then create a PR on github with intelligent commit messages based on staged changes. Invoke with /create-pr.
---

# Create Pull Request

Rebase from the latest `origin/main`, squash commits, and create a PR on GitHub with an
intelligent title and description.

## Usage

```
/create-pr [--draft] [--base <branch>]
```

**Arguments:**

- `--draft`: Create as draft PR
- `--base <branch>`: Target branch (default: `main`)

## Workflow

### Step 1: Verify Prerequisites

```bash
# Check current branch
git branch --show-current

# Check if on main (should NOT be)
if [[ $(git branch --show-current) == "main" ]]; then
  echo "ERROR: Cannot create PR from main branch"
  exit 1
fi

# Check for uncommitted changes
git status --short

# Ensure gh CLI is available
gh --version
```

**Action:** If there are uncommitted changes, stop, and then ask user to commit or stash
them first.

### Step 2: Check for Existing PR

```bash
# Check if PR already exists for current branch
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

**Handle Existing PR:**

- If PR exists, inform user and ask permission to force-update it
- Warn that this will rewrite the commit history and PR description
- If user declines, abort the process

### Step 3: Fetch and Rebase

```bash
# Fetch latest from origin
git fetch origin main

# Check divergence
git log --oneline HEAD ^origin/main

# Non-interactive rebase onto origin/main
git rebase origin/main
```

**Handle Conflicts:** If rebase fails due to conflicts, abort and let user handle rebase
manually:

```bash
git rebase --abort
echo "Rebase failed due to conflicts. Please resolve manually and retry /create-pr"
exit 1
```

### Step 4: Squash Commits into Single Commit

After successful rebase, squash all commits since `origin/main` into a single commit:

```bash
# Count commits to squash
git rev-list --count origin/main..HEAD

# Soft reset to origin/main (keeps changes staged)
git reset --soft origin/main
```

Generate commit message following `/gen-commit-msg` format.

### Step 5: Analyze Combined Changes

```bash
# Get all changes since origin/main
git diff origin/main...HEAD --name-only

# Get full diff content
git diff origin/main...HEAD
```

**Determine Scope** from changed files:

- `calibration/` → `calibration`
- `pose_6d/` → `pose_6d`
- `gripper/` → `gripper`
- `core/` → `core`
- `runners/` → `runners`
- `ros2_ws/` → `ros2`
- `scripts/` → `scripts`
- `config/` → `config`
- `pixi.toml` → `deps`
- Multiple areas → omit scope or use broader term

### Step 6: Generate PR Title and Description

**PR Title Format:**

```
<type>(<scope>): <brief description>
```

**Rules:**

- Keep under 70 characters
- Use imperative mood
- No period at end

**PR Description Format:**

```markdown
## Description

[2-4 sentences explaining what this PR does and why]

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing

- [ ] Ran linter / formatter if configured
- [ ] Replayed pipeline on the default test video (`data/aruco_test/VID_*.insv`)
- [ ] Verified viser overlay looks correct (if visualization changed)
- [ ] Rebuilt `ros2_ws` if C++ / CMakeLists touched

## Run commands

If the PR changes runnable behavior, include the exact command(s) used to
verify it so a reviewer can reproduce. Use the default test video /
calibration paths when applicable. Example:

```
pixi run python -m runners.rig_replay \
    --video data/aruco_test/VID_20260518_093406_00_010.insv \
    --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \
    --world-marker-configs config/aruco_bench.yaml \
    --hinge-marker-config config/chopsticks-v1.yaml
```

## Files Changed

- `path/to/file.py`: Description of change

## Additional Context

[Any extra info, hardware requirements, related issues]
```

### Step 7: Push and Create/Update PR

Show preview to user, then after confirmation:

```bash
# Force push branch to remote (required after squash)
git push -f -u origin $(git branch --show-current)

# Create or edit PR using gh CLI
if gh pr view &>/dev/null; then
  gh pr edit \
    --title "<title>" \
    --body "$(cat <<'EOF'
[PR description here]
EOF
)"
else
  gh pr create \
    --base main \
    --title "<title>" \
    --body "$(cat <<'EOF'
[PR description here]
EOF
)"
fi
```

Add `--draft` flag if requested.

**Capture PR URL** and display to user:

```
✓ PR created/updated successfully!
https://github.com/xxx/xxx/pull/123
```

## Error Handling

### Rebase Conflicts

If rebase fails:

1. Show conflict files
1. Provide resolution instructions
1. Wait for user to resolve
1. After resolution, continue with squashing step
1. Offer to abort rebase if needed: `git rebase --abort`

### Squash Failures

If squash/commit fails:

1. Check if there are changes to commit: `git status`
1. Verify no conflicts remain: `git diff --cached`
1. If needed, abort and return to pre-rebase state

### Push Failures

If force push fails:

1. Verify remote branch exists
1. Check GitHub authentication: `gh auth status`
1. Confirm branch protection rules allow force push
1. Provide manual push instructions if needed

### PR Creation/Update Failures

If `gh pr create` or `gh pr edit` fails:

1. Check if PR already exists: `gh pr view`
1. Verify GitHub authentication: `gh auth status`
1. Check for branch protection rules
1. Provide manual PR creation/update link


## Safety Checks

**Before Starting:**

- Confirm no uncommitted changes
- Confirm not on main/main branch
- Check for existing PR and get user permission to overwrite if exists
- Backup branch: `git branch backup/$(git branch --show-current)-$(date +%s)`

**Before Rebase:**

- Fetch latest from origin
- Show divergence summary

**Before Squash:**

- Show commits that will be squashed
- Confirm user wants to proceed

**Before Force Push:**

- **CRITICAL**: Warn user that force push will rewrite history
- Show current commit that will replace remote history
- Confirm branch name
- If PR exists, emphasize that PR history will be rewritten

**Before PR Creation/Update:**

- Show full preview of title/description
- Confirm target branch
- If updating existing PR, show what will change

## Examples

### Example 1: Feature PR

**Changes:** Add hinge-angle estimation to the gripper pipeline

**PR Title:**

```
feat(gripper): add hinge-angle estimation from dot pairs
```

**PR Description:**

```markdown
## Description

Compute hinge angle from two tracked dots on the gripper jaw using the
plane-space helpers in `core/geometry`. Wires the new `HingeAngleEstimator`
into `RigPipeline` so `runners/rig_replay.py` overlays the angle in viser
alongside the 6D pose.

## Type of Change

- [ ] Bug fix
- [x] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing

- [x] Ran linter / formatter if configured
- [x] Replayed pipeline on the default test video (`data/aruco_test/VID_*.insv`)
- [x] Verified viser overlay looks correct (if visualization changed)
- [ ] Rebuilt `ros2_ws` if C++ / CMakeLists touched

## Run commands

```
pixi run python -m runners.rig_replay \
    --video data/aruco_test/VID_20260518_093406_00_010.insv \
    --intrinsics data/insta360_calibration/lens0_combined_subpixel_best.npz \
    --world-marker-configs config/aruco_bench.yaml \
    --hinge-marker-config config/chopsticks-v1.yaml
```

## Files Changed

- `gripper/estimator.py`: New `HingeAngleEstimator` class
- `gripper/dots.py`: Dot-pair detection helpers built on `core/markers`
- `gripper/pipeline.py`: Wire estimator into `RigPipeline`
- `runners/rig_replay.py`: Add hinge-angle overlay
- `core/geometry.py`: Expose `plane_angle()` helper

## Additional Context

Uses the default test video. Falls back to NaN when only one dot is visible
so downstream consumers can skip frames gracefully.
```

### Example 2: Bug Fix PR

**Changes:** Fix fisheye undistortion returning mirrored points

**PR Title:**

```
fix(calibration): correct sign on fisheye undistortion
```

**PR Description:**

```markdown
## Description

Fix mirrored output from `fisheye.undistort_points()`: the inverse Brown-Conrady
step was applying `-k1` instead of `+k1`, which flipped points across the
principal axis. The error only showed up on wide-FoV lenses where the radial
term dominates. Adds a regression check on the lens0 calibration npz.

## Type of Change

- [x] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring
- [ ] Performance improvement

## Testing

- [x] Ran linter / formatter if configured
- [x] Replayed pipeline on the default test video (`data/aruco_test/VID_*.insv`)
- [x] Verified viser overlay looks correct (if visualization changed)
- [ ] Rebuilt `ros2_ws` if C++ / CMakeLists touched

## Files Changed

- `calibration/fisheye.py`: Flip sign on inverse radial term
- `scripts/bench/check_undistort.py`: Regression check against calibrated npz

## Additional Context

Verified against the lens0 sub-pixel calibration: undistorted ArUco corners
now project back onto their detected pixel positions to within 0.3 px.
```

### Example 3: Breaking Change PR

**Changes:** Split monolithic pose calibration into per-stage subpackages

**PR Title:**

```
refactor(pose_6d): split pose_calibration into stage subpackages
```

**PR Description:**

```markdown
## Description

Replace the monolithic `pose_calibration` module with `calibration/`,
`pose_6d/`, and shared `core/` packages. Each stage now exposes its own
public surface; callers that previously imported from `pose_calibration.*`
need to update import paths. Runner entry points moved to `runners/` and
the old `pose_calibration` package is removed.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [x] Breaking change
- [ ] Documentation update
- [x] Refactoring
- [ ] Performance improvement

## Testing

- [x] Ran linter / formatter if configured
- [x] Replayed pipeline on the default test video (`data/aruco_test/VID_*.insv`)
- [x] Verified viser overlay looks correct (if visualization changed)
- [ ] Rebuilt `ros2_ws` if C++ / CMakeLists touched

## Files Changed

- `calibration/`: New package for fisheye / pinhole / two-stage calibration
- `pose_6d/`: New package for estimator, known-board, layout, learned-layout
- `core/`: Shared infrastructure (camera, geometry, markers, viz)
- `runners/replay_insta.py`, `runners/replay_video.py`: Moved from old module
- `pose_calibration/`: **Removed**

## Additional Context

**Breaking change**: `from pose_calibration import X` now fails. Update to the
new package paths (e.g., `from pose_6d import PoseEstimator`). No on-disk
formats changed, so cached calibration npz files still load.
```

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/commands/create-pr.md
Invocation: /create-pr

## Design Philosophy

- Automates full PR creation workflow: fetch, rebase, **squash to single commit**, push, create/update PR
- **Always squashes all commits** since `origin/main` into a single commit with message generated via `/gen-commit-msg` logic
- **Handles existing PRs** by detecting them and force-updating after user permission
- Follows repository's Conventional Commits format
- Requires user confirmation at critical steps (existing PR detection, rebase, squash, force-push, PR creation/update)
- Generates intelligent commit messages, PR titles, and descriptions based on change analysis
- Uses force push (`-f`) by design, as squashing requires rewriting history

## How to Update

### Adding New Scopes
Update "Determine Scope" section with new file path mappings.

### Changing PR Template
Update "PR Description Format" section with new template structure.

### Modifying Workflow Steps
Update relevant "Step N" sections with new git commands or logic.

================================================================================
-->

