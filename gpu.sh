#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
gpu.sh: interactive menu to list GPU-using PIDs (PID, app name, VRAM in GB) and kill them

Usage:
  ./gpu.sh

Requires: nvidia-smi (NVIDIA driver)
EOF
}

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }

is_windows_bash() {
  # Git Bash / MSYS2 / Cygwin: nvidia-smi reports Windows PIDs; `kill` targets MSYS PIDs.
  case "$(uname -s 2>/dev/null || true)" in
    MINGW*|MSYS*|CYGWIN*) return 0;;
    *) return 1;;
  esac
}

trim() { sed -e 's/^[[:space:]]\+//' -e 's/[[:space:]]\+$//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

need nvidia-smi

list_apps_raw() {
  # Prefer both compute + graphics if available. Some driver modes only support one.
  (nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  (nvidia-smi --query-graphics-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
}

list_apps() {
  # Output: pid<TAB>name<TAB>mib
  list_apps_raw \
    | awk -F',' '{
        for (i=1;i<=NF;i++) { gsub(/^[ \t]+|[ \t]+$/, "", $i) }
        pid=$1; name=$2; mib=$3
        if (pid=="" || pid=="No running processes found") next
        if (pid !~ /^[0-9]+$/) next
        if (mib=="") mib=0
        # If a PID appears in both compute and graphics queries, keep the max MiB.
        key=pid "\t" name
        if (mib+0 > mem[key]+0) mem[key]=mib+0
      }
      END{
        for (k in mem) {
          split(k, a, "\t");
          printf "%s\t%s\t%d\n", a[1], a[2], mem[k]
        }
      }' \
    | sort -k3,3nr -k1,1n
}

do_kill_pid() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || { echo "Not a PID: $pid" >&2; return 2; }

  if is_windows_bash; then
    need taskkill
    echo "taskkill /PID $pid /T"
    if taskkill /PID "$pid" /T >/dev/null 2>&1; then
      return 0
    fi
    echo "PID $pid: taskkill failed. Force kill? [y/N] "
    read -r ans
    case "${ans:-}" in
      y|Y)
        echo "taskkill /PID $pid /T /F"
        taskkill /PID "$pid" /T /F >/dev/null 2>&1 || { echo "PID $pid: not found (already exited?)" >&2; return 1; }
        ;;
      *)
        echo "Skipped PID $pid" >&2
        return 1
        ;;
    esac
    return 0
  fi

  echo "kill -TERM $pid"
  if ! kill -TERM "$pid" 2>/dev/null; then
    echo "PID $pid: not found (already exited?)" >&2
    return 1
  fi
  sleep 0.2
  if kill -0 "$pid" 2>/dev/null; then
    echo "PID $pid still running. Force kill? [y/N] "
    read -r ans
    case "${ans:-}" in
      y|Y)
        echo "kill -KILL $pid"
        kill -KILL "$pid" 2>/dev/null || true
        ;;
    esac
  fi
}

print_screen() {
  command -v clear >/dev/null 2>&1 && clear || true
  echo "GPU Processes (from nvidia-smi)"
  echo
  local rows
  rows="$(list_apps)"
  if [[ -z "${rows//[[:space:]]/}" ]]; then
    echo "No GPU-using processes found."
  else
    printf "%-8s %-10s %s\n" "PID" "VRAM_GB" "APP"
    echo "$rows" | awk -F'\t' '{ printf "%-8s %-10.2f %s\n", $1, ($3/1024.0), $2 }'
  fi
  echo
  echo "Options: [K]ill PID  Kill [A]ll  [R]efresh  [Q]uit"
}

confirm() {
  local prompt="$1"
  echo "$prompt [y/N] "
  read -r ans
  case "${ans:-}" in
    y|Y) return 0;;
    *) return 1;;
  esac
}

kill_all() {
  mapfile -t pids < <(list_apps | awk -F'\t' '{print $1}' | sort -n | uniq)
  if (( ${#pids[@]} == 0 )); then
    echo "No GPU-using PIDs found."
    return 0
  fi
  if ! confirm "Kill ALL listed PIDs?"; then
    return 0
  fi
  for pid in "${pids[@]}"; do
    do_kill_pid "$pid" || true
  done
}

while true; do
  print_screen
  read -r -p "> " choice || exit 0
  case "${choice:-}" in
    q|Q|exit|EXIT) exit 0;;
    r|R|"") continue;;
    a|A)
      kill_all
      echo
      read -r -p "Press Enter to continue..." _ || true
      ;;
    k|K)
      read -r -p "PID to kill: " pid
      if [[ -z "${pid:-}" ]]; then
        continue
      fi
      if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
        echo "Not a PID: $pid" >&2
        read -r -p "Press Enter to continue..." _ || true
        continue
      fi
      if confirm "Kill PID $pid?"; then
        do_kill_pid "$pid" || true
      fi
      echo
      read -r -p "Press Enter to continue..." _ || true
      ;;
    *)
      echo "Unknown option: $choice" >&2
      read -r -p "Press Enter to continue..." _ || true
      ;;
  esac
done
