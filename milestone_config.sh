#!/usr/bin/env bash
# ==============================================================================
# milestone_config.sh
#
# Sourced by sweep_intermediate.sh. One entry per STEP-COUNT (shared across
# solver, dataset, and length_range -- exactly as requested: you only set
# this once per step count, not per combination).
#
# Each value is EITHER:
#   - a plain integer  -> passed as --num_milestones N (evenly spaced + final)
#   - a comma-separated list -> passed as --milestone_steps "a,b,c" (exact
#     calls-remaining labels you name yourself; see intermediate_sample.py's
#     own docstring for how these labels map to real model calls)
#
# Edit the values on the right freely; sweep_intermediate.sh re-sources this
# file on every run, so changes take effect on the next invocation.
# ==============================================================================

declare -A MILESTONES

# ancestral is its own entry (2000 real model calls -- the full trajectory)
MILESTONES[2000]="20"

# dpm_solver / ddim step counts -- same milestone spec used for BOTH solvers
# at a given step count, per your spec ("milestones will be the same across
# solvers"). Edit freely, e.g. MILESTONES[10]="7,6,2" for an explicit list.
MILESTONES[500]="10"
MILESTONES[200]="10"
MILESTONES[100]="10"
MILESTONES[10]="10"
MILESTONES[5]="5"
MILESTONES[4]="4"
MILESTONES[2]="2"