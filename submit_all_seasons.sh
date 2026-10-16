#!/bin/bash
# =========================================================
# Submit All 4 Seasonal PBS Jobs to Cluster Queue
# =========================================================

set -e

echo "========================================================="
echo "Submitting 4 Seasonal Optimization Jobs to PBS Queue"
echo "========================================================="

echo -n "Submitting Winter Job... "
WINTER_ID=$(qsub job_winter.pbs)
echo "JobID: $WINTER_ID"

echo -n "Submitting Summer Job... "
SUMMER_ID=$(qsub job_summer.pbs)
echo "JobID: $SUMMER_ID"

echo -n "Submitting Monsoon Job... "
MONSOON_ID=$(qsub job_monsoon.pbs)
echo "JobID: $MONSOON_ID"

echo -n "Submitting Post-Monsoon Job... "
POST_ID=$(qsub job_post.pbs)
echo "JobID: $POST_ID"

echo "========================================================="
echo "All 4 jobs successfully submitted!"
echo "Check queue status with: qstat -u \$USER"
echo "========================================================="
