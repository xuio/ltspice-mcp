#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_WORKDIR="${PROJECT_ROOT}/.mcp-workdir"
DAEMON_DIR="${LTSPICE_MCP_DAEMON_DIR:-${DEFAULT_WORKDIR}/daemon}"
LOG_DIR="${DAEMON_DIR}/logs"
PID_FILE="${DAEMON_DIR}/ltspice-mcp-daemon.pid"
LATEST_LOG_LINK="${LOG_DIR}/latest.log"

HOST="${LTSPICE_MCP_DAEMON_HOST:-127.0.0.1}"
PORT="${LTSPICE_MCP_DAEMON_PORT:-8765}"
HTTP_PATH="${LTSPICE_MCP_DAEMON_HTTP_PATH:-/mcp}"
WORKDIR="${LTSPICE_MCP_DAEMON_WORKDIR:-${DEFAULT_WORKDIR}}"
TIMEOUT="${LTSPICE_MCP_DAEMON_TIMEOUT:-180}"
LTSPICE_BINARY="${LTSPICE_MCP_DAEMON_LTSPICE_BINARY:-/Applications/LTspice.app/Contents/MacOS/LTspice}"
UV_BIN="${UV_BIN:-uv}"

if [[ "${HTTP_PATH}" != /* ]]; then
  HTTP_PATH="/${HTTP_PATH}"
fi

mkdir -p "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage: ./scripts/ltspice_mcp_daemon.sh <command> [options]

Commands:
  start                 Start the LTspice MCP daemon (HTTP mode via uv)
  stop                  Stop the daemon
  restart               Restart the daemon
  status                Print daemon status
  follow [--lines N]    Follow daemon logs in real time
  logs [--lines N]      Print latest daemon log output (default: 120 lines)
  logs --follow         Follow the latest daemon log
  latest-log            Print absolute path to the latest daemon log file
  list-logs [N]         List newest daemon log files (default: 5)

Environment overrides:
  LTSPICE_MCP_DAEMON_HOST
  LTSPICE_MCP_DAEMON_PORT
  LTSPICE_MCP_DAEMON_HTTP_PATH
  LTSPICE_MCP_DAEMON_WORKDIR
  LTSPICE_MCP_DAEMON_TIMEOUT
  LTSPICE_MCP_DAEMON_LTSPICE_BINARY
  LTSPICE_MCP_DAEMON_DIR
  UV_BIN
EOF
}

url() {
  echo "http://${HOST}:${PORT}${HTTP_PATH}"
}

pid_from_file() {
  [[ -f "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  echo "${pid}"
}

is_ltspice_daemon_pid() {
  local pid="$1"
  local cmd
  cmd="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
  [[ "${cmd}" == *"ltspice-mcp"* && "${cmd}" == *"--daemon-http"* ]]
}

daemon_pids_from_port() {
  local pid
  for pid in $(lsof -nP -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true); do
    if is_ltspice_daemon_pid "${pid}"; then
      echo "${pid}"
    fi
  done
}

daemon_pids_from_ps() {
  pgrep -f "ltspice-mcp --daemon-http --host ${HOST} --port ${PORT}" 2>/dev/null || true
}

foreign_pids_from_port() {
  local pid
  for pid in $(lsof -nP -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true); do
    if ! is_ltspice_daemon_pid "${pid}"; then
      echo "${pid}"
    fi
  done
}

all_known_daemon_pids() {
  {
    pid_from_file 2>/dev/null || true
    daemon_pids_from_port || true
    daemon_pids_from_ps || true
  } | awk 'NF' | sort -u
}

latest_log_path() {
  local latest=""
  if [[ -L "${LATEST_LOG_LINK}" ]]; then
    local linked
    linked="$(readlink "${LATEST_LOG_LINK}" || true)"
    if [[ -n "${linked}" ]]; then
      if [[ "${linked}" == /* ]]; then
        latest="${linked}"
      else
        latest="${LOG_DIR}/${linked}"
      fi
    fi
  elif [[ -f "${LATEST_LOG_LINK}" ]]; then
    latest="${LATEST_LOG_LINK}"
  fi

  if [[ -z "${latest}" || ! -f "${latest}" ]]; then
    latest="$(ls -1t "${LOG_DIR}"/ltspice-mcp-daemon-*.log 2>/dev/null | head -n 1 || true)"
  fi

  [[ -n "${latest}" && -f "${latest}" ]] && echo "${latest}"
}

is_running() {
  local pid
  pid="$(pid_from_file 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi

  pid="$(daemon_pids_from_port | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${PID_FILE}"
    return 0
  fi

  pid="$(daemon_pids_from_ps | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${PID_FILE}"
    return 0
  fi

  return 1
}

start_daemon() {
  if is_running; then
    local pid
    pid="$(pid_from_file)"
    echo "ltspice-mcp daemon already running (pid ${pid})"
    echo "URL: $(url)"
    return 0
  fi

  local foreign
  foreign="$(foreign_pids_from_port || true)"
  if [[ -n "${foreign}" ]]; then
    echo "Port ${PORT} is in use by non-ltspice process(es): ${foreign}" >&2
    echo "Pick a different port or stop those processes first." >&2
    return 1
  fi

  local timestamp log_file pid uv_pid listener_pid
  timestamp="$(date '+%Y%m%d-%H%M%S')"
  log_file="${LOG_DIR}/ltspice-mcp-daemon-${timestamp}.log"

  (
    cd "${PROJECT_ROOT}"
    nohup "${UV_BIN}" run --project "${PROJECT_ROOT}" ltspice-mcp \
      --daemon-http \
      --host "${HOST}" \
      --port "${PORT}" \
      --http-path "${HTTP_PATH}" \
      --workdir "${WORKDIR}" \
      --timeout "${TIMEOUT}" \
      --ltspice-binary "${LTSPICE_BINARY}" \
      > "${log_file}" 2>&1 &
    echo $! > "${PID_FILE}"
  )

  ln -sfn "$(basename "${log_file}")" "${LATEST_LOG_LINK}"
  uv_pid="$(cat "${PID_FILE}")"
  listener_pid=""

  local tries=0
  while [[ ${tries} -lt 40 ]]; do
    listener_pid="$(daemon_pids_from_port | head -n 1 || true)"
    if [[ -n "${listener_pid}" ]]; then
      break
    fi

    if ! kill -0 "${uv_pid}" 2>/dev/null; then
      break
    fi

    tries=$((tries + 1))
    sleep 0.5
  done

  if [[ -n "${listener_pid}" ]]; then
    echo "${listener_pid}" > "${PID_FILE}"
    pid="${listener_pid}"
  elif kill -0 "${uv_pid}" 2>/dev/null; then
    echo "${uv_pid}" > "${PID_FILE}"
    pid="${uv_pid}"
  else
    pid=""
  fi

  if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
    echo "Failed to start daemon. Recent log output:" >&2
    tail -n 120 "${log_file}" >&2 || true
    rm -f "${PID_FILE}"
    stop_daemon >/dev/null 2>&1 || true
    return 1
  fi

  echo "Started ltspice-mcp daemon (pid ${pid})"
  echo "URL: $(url)"
  echo "Log: ${log_file}"
}

stop_daemon() {
  local pids
  pids="$(all_known_daemon_pids)"
  if [[ -z "${pids}" ]]; then
    rm -f "${PID_FILE}"
    echo "ltspice-mcp daemon is not running"
    return 0
  fi

  local pid
  for pid in ${pids}; do
    kill "${pid}" 2>/dev/null || true
  done

  local tries=0
  while [[ ${tries} -lt 40 ]]; do
    local remaining=""
    for pid in ${pids}; do
      if kill -0 "${pid}" 2>/dev/null; then
        remaining="${remaining} ${pid}"
      fi
    done
    if [[ -z "${remaining// }" ]]; then
      rm -f "${PID_FILE}"
      echo "Stopped ltspice-mcp daemon"
      return 0
    fi
    tries=$((tries + 1))
    sleep 0.25
  done

  for pid in ${pids}; do
    kill -9 "${pid}" 2>/dev/null || true
  done
  rm -f "${PID_FILE}"
  echo "Force-stopped ltspice-mcp daemon"
}

status_daemon() {
  if is_running; then
    local pid log_file
    pid="$(pid_from_file)"
    log_file="$(latest_log_path || true)"
    echo "ltspice-mcp daemon: running (pid ${pid})"
    echo "URL: $(url)"
    if [[ -n "${log_file}" ]]; then
      echo "Latest log: ${log_file}"
    fi
  else
    echo "ltspice-mcp daemon: not running"
    echo "URL: $(url)"
    return 1
  fi
}

logs_daemon() {
  local follow=false
  local lines=120

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --follow|-f)
        follow=true
        shift
        ;;
      --lines|-n)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for $1" >&2
          return 1
        fi
        lines="$2"
        shift 2
        ;;
      *)
        echo "Unknown option for logs: $1" >&2
        return 1
        ;;
    esac
  done

  if ! [[ "${lines}" =~ ^[0-9]+$ ]]; then
    echo "--lines must be an integer" >&2
    return 1
  fi

  local log_file
  log_file="$(latest_log_path || true)"
  if [[ -z "${log_file}" ]]; then
    echo "No daemon log file found in ${LOG_DIR}" >&2
    return 1
  fi

  if [[ "${follow}" == true ]]; then
    if [[ -L "${LATEST_LOG_LINK}" || -f "${LATEST_LOG_LINK}" ]]; then
      tail -n "${lines}" -F "${LATEST_LOG_LINK}"
    else
      tail -n "${lines}" -F "${log_file}"
    fi
  else
    tail -n "${lines}" "${log_file}"
  fi
}

follow_daemon() {
  logs_daemon --follow "$@"
}

list_logs() {
  local count="${1:-5}"
  if ! [[ "${count}" =~ ^[0-9]+$ ]]; then
    echo "list-logs count must be an integer" >&2
    return 1
  fi

  ls -1t "${LOG_DIR}"/ltspice-mcp-daemon-*.log 2>/dev/null | head -n "${count}" || true
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 1
fi
shift || true

case "${command}" in
  start)
    start_daemon
    ;;
  stop)
    stop_daemon
    ;;
  restart)
    stop_daemon
    start_daemon
    ;;
  status)
    status_daemon
    ;;
  follow)
    follow_daemon "$@"
    ;;
  logs)
    logs_daemon "$@"
    ;;
  latest-log)
    latest_log_path
    ;;
  list-logs)
    list_logs "${1:-5}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
