#!/usr/bin/env bash
# Run an OpenRouter sweep on a GCE VM instead of a workstation.
#
# The sweep is API-bound, not CPU-bound: it spends its life waiting on provider
# HTTP responses, so a small VM is as fast as a large one and the only thing that
# matters is that it can stay up for hours without competing for a laptop.
#
# Usage:
#   gcloud auth login                      # once, interactively
#   scripts/run_sweep_on_gce.sh up         # create VM, upload, start sweep
#   scripts/run_sweep_on_gce.sh status     # progress + spend
#   scripts/run_sweep_on_gce.sh logs       # tail the sweep log
#   scripts/run_sweep_on_gce.sh fetch      # copy results back into the repo
#   scripts/run_sweep_on_gce.sh down       # delete the VM
#
# The sweep is resumable: `up` on an existing VM restarts it and it skips models
# already recorded in raw_summary.csv.

set -euo pipefail

VM="${OPENVOICECS_VM:-openvoicecs-sweep}"
ZONE="${OPENVOICECS_ZONE:-us-central1-a}"
MACHINE="${OPENVOICECS_MACHINE:-e2-standard-4}"
RUN_ID="${OPENVOICECS_RUN_ID:-openrouter_top50_text_action_3trial_v02}"
TRIALS="${OPENVOICECS_TRIALS:-3}"
TOP="${OPENVOICECS_TOP:-50}"
WORKERS="${OPENVOICECS_WORKERS:-16}"
# Point at a different ranked model table to sweep a specific shortlist, e.g.
# OPENVOICECS_MODELS_FILE=data/openvoicecs/requested_models_v02.md
MODELS_FILE="${OPENVOICECS_MODELS_FILE:-openrouter-top-100-models.md}"
REMOTE_DIR="openvoicecs-bench"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_auth() {
  if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "gcloud auth has expired. Run: gcloud auth login" >&2
    exit 1
  fi
}

# Provisioning defaults to STANDARD. Spot saves a few cents an hour on a job that
# costs well under a dollar, and a preemption mid-sweep means nobody can walk
# away from it — which is the entire point of running it off a workstation. Set
# OPENVOICECS_SPOT=1 if you want the discount and can babysit restarts.
cmd_up() {
  require_auth
  if ! gcloud compute instances describe "$VM" --zone "$ZONE" >/dev/null 2>&1; then
    local provisioning=(--provisioning-model STANDARD)
    if [ "${OPENVOICECS_SPOT:-0}" = "1" ]; then
      provisioning=(--provisioning-model SPOT --instance-termination-action STOP)
    fi
    echo "==> creating $VM ($MACHINE, ${OPENVOICECS_SPOT:+spot}${OPENVOICECS_SPOT:-standard}) in $ZONE"
    gcloud compute instances create "$VM" \
      --zone "$ZONE" \
      --machine-type "$MACHINE" \
      --image-family debian-12 \
      --image-project debian-cloud \
      --boot-disk-size 50GB \
      "${provisioning[@]}" \
      --scopes cloud-platform \
      --labels purpose=openvoicecs-sweep
    echo "==> waiting for ssh"
    until gcloud compute ssh "$VM" --zone "$ZONE" --command true >/dev/null 2>&1; do sleep 5; done
  else
    gcloud compute instances start "$VM" --zone "$ZONE" >/dev/null 2>&1 || true
    echo "==> waiting for ssh"
    until gcloud compute ssh "$VM" --zone "$ZONE" --command true >/dev/null 2>&1; do sleep 5; done
  fi

  echo "==> installing python"
  gcloud compute ssh "$VM" --zone "$ZONE" --command \
    'sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-venv python3-pip tmux rsync >/dev/null'

  # Historical runs are ~125 MB of past results the sweep never reads. They are
  # excluded by name, which leaves the in-progress run for THIS run-id in the
  # payload so the sweep resumes from its checkpoint instead of re-paying for
  # models already scored.
  echo "==> uploading repo (excluding historical results)"
  gcloud compute ssh "$VM" --zone "$ZONE" --command "mkdir -p ~/$REMOTE_DIR"
  COPYFILE_DISABLE=1 tar --exclude='.git' \
      --exclude='data/openvoicecs/runs/openrouter_top50_text_action_3trial_20260615' \
      --exclude='data/openvoicecs/runs/openrouter_top50_text_action_3trial_actionloop_20260615' \
      --exclude='data/openvoicecs/runs/openrouter_top50_text_action_3trial_jsontrace_20260615' \
      --exclude='data/openvoicecs/reports' \
      --exclude='.venv' --exclude='venv' \
      --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.ruff_cache' \
      -czf /tmp/openvoicecs-payload.tgz -C "$REPO_ROOT" .
  gcloud compute scp /tmp/openvoicecs-payload.tgz "$VM":~/payload.tgz --zone "$ZONE"
  gcloud compute ssh "$VM" --zone "$ZONE" --command \
    "tar -xzf ~/payload.tgz -C ~/$REMOTE_DIR && rm ~/payload.tgz"
  rm -f /tmp/openvoicecs-payload.tgz

  echo "==> installing package"
  gcloud compute ssh "$VM" --zone "$ZONE" --command \
    "cd ~/$REMOTE_DIR && python3 -m venv .venv && ./.venv/bin/pip install -q -e '.[dev,providers]'"

  echo "==> starting sweep in tmux (resumable)"
  gcloud compute ssh "$VM" --zone "$ZONE" --command \
    "cd ~/$REMOTE_DIR && tmux kill-session -t sweep 2>/dev/null || true; \
     tmux new-session -d -s sweep \
     'set -a; . ./.env; set +a; ./.venv/bin/python scripts/run_openrouter_top50_text_action_batch.py \
        --models-file $MODELS_FILE \
        --top $TOP --trials $TRIALS --workers $WORKERS --skip-judging \
        --adapter-mode json-trace --model-timeout-seconds 5400 --min-credits-usd 5 \
        --run-id $RUN_ID --output-root data/openvoicecs/runs 2>&1 | tee -a ~/sweep.log'"
  echo "==> running. scripts/run_sweep_on_gce.sh status"
}

cmd_status() {
  require_auth
  gcloud compute ssh "$VM" --zone "$ZONE" --command \
    "cd ~/$REMOTE_DIR && echo \"models done: \$(ls data/openvoicecs/runs/$RUN_ID/reports 2>/dev/null | wc -l)/$TOP\" && \
     echo \"tmux: \$(tmux has-session -t sweep 2>/dev/null && echo running || echo stopped)\" && \
     cut -d, -f1,3 data/openvoicecs/runs/$RUN_ID/raw_summary.csv 2>/dev/null | tail -20"
}

cmd_logs() {
  require_auth
  gcloud compute ssh "$VM" --zone "$ZONE" --command \
    "grep -vE 'HTTP Request|INFO' ~/sweep.log | tail -40"
}

cmd_fetch() {
  require_auth
  echo "==> fetching results into $REPO_ROOT/data/openvoicecs/runs/"
  mkdir -p "$REPO_ROOT/data/openvoicecs/runs"
  gcloud compute scp --recurse \
    "$VM":"~/$REMOTE_DIR/data/openvoicecs/runs/$RUN_ID" \
    "$REPO_ROOT/data/openvoicecs/runs/" --zone "$ZONE"
  echo "==> building leaderboard"
  python "$REPO_ROOT/scripts/build_leaderboard.py" \
    "$REPO_ROOT/data/openvoicecs/runs/$RUN_ID/reports" \
    --output "$REPO_ROOT/data/openvoicecs/runs/$RUN_ID/leaderboard.csv"
}

cmd_down() {
  require_auth
  gcloud compute instances delete "$VM" --zone "$ZONE" --quiet
}

case "${1:-}" in
  up) cmd_up ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  fetch) cmd_fetch ;;
  down) cmd_down ;;
  *) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
