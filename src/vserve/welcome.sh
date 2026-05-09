#!/bin/bash
# vserve login banner — reads from vserve config (single source of truth)
# Installed by: vserve init → ~/.config/vserve/welcome.sh
# Remove with:  rm ~/.config/vserve/welcome.sh  (and delete the source line from your shell rc)

[ -z "$BASH_VERSION" ] && [ -z "$ZSH_VERSION" ] && return 0
command -v gum &>/dev/null || return 0

_esc="$(printf '\033')"
_reset="${_esc}[0m"
_c() { printf '%s[38;5;%sm%s%s' "$_esc" "$1" "$2" "$_reset"; }
_cb() { printf '%s[1;38;5;%sm%s%s' "$_esc" "$1" "$2" "$_reset"; }

# ── Read vserve config ──────────────────────────────────

_vserve_cfg="$HOME/.config/vserve/config.yaml"
_vllm_root="/opt/vllm"
_svc_name="vllm"
_port="8888"

_llamacpp_root=""
_llamacpp_svc="llama-cpp"

if [ -f "$_vserve_cfg" ]; then
    _cfg_values="$(python3 - "$_vserve_cfg" 2>/dev/null <<'PY'
import shlex
import sys

import yaml

with open(sys.argv[1]) as f:
    data = yaml.safe_load(f) or {}
backends = data.get("backends") or {}
vllm = backends.get("vllm") or {}
llamacpp = backends.get("llamacpp") or {}

def emit(name, value):
    if value is not None and value != "":
        print(f"{name}={shlex.quote(str(value))}")

emit("_vllm_root", vllm.get("root", data.get("vllm_root")))
emit("_svc_name", vllm.get("service_name", data.get("service_name")))
emit("_port", vllm.get("port", data.get("port")))
emit("_llamacpp_root", llamacpp.get("root", data.get("llamacpp_root")))
emit("_llamacpp_svc", llamacpp.get("service_name", data.get("llamacpp_service_name")))
PY
)"
    if [ -n "$_cfg_values" ]; then
        eval "$_cfg_values"
    else
        _vllm_root="$(grep '^vllm_root:' "$_vserve_cfg" 2>/dev/null | sed 's/vllm_root: *//')" || _vllm_root="/opt/vllm"
        _svc_name="$(grep '^service_name:' "$_vserve_cfg" 2>/dev/null | sed 's/service_name: *//')" || _svc_name="vllm"
        _port="$(grep '^port:' "$_vserve_cfg" 2>/dev/null | sed 's/port: *//')" || _port="8888"
        _llamacpp_root="$(grep '^llamacpp_root:' "$_vserve_cfg" 2>/dev/null | sed 's/llamacpp_root: *//')"
    fi
fi

# ── Gather data ──────────────────────────────────────

_s="inactive"
_active_backend=""
if systemctl is-active --quiet "$_svc_name" 2>/dev/null; then
    _s="active"
    _active_backend="vllm"
elif [ -n "$_llamacpp_root" ] && systemctl is-active --quiet "$_llamacpp_svc" 2>/dev/null; then
    _s="active"
    _active_backend="llamacpp"
fi

_m=""
if [ "$_active_backend" = "vllm" ]; then
    _active="$_vllm_root/configs/active.yaml"
    if [ -f "$_active" ]; then
        _m="$(grep '^model:' "$_active" 2>/dev/null | sed "s|model: *${_vllm_root}/models/||; s|model: *||")"
    fi
elif [ "$_active_backend" = "llamacpp" ]; then
    if [ -f "$_llamacpp_root/configs/active.json" ]; then
        _m="$(grep -o '"model" *: *"[^"]*"' "$_llamacpp_root/configs/active.json" 2>/dev/null | sed 's/"model" *: *"//;s/"$//' | sed 's|.*/||')"
    fi
fi

_mc=0
if [ -d "$_vllm_root/models" ]; then
    _mc="$(/usr/bin/find "$_vllm_root/models" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)"
fi
if [ -n "$_llamacpp_root" ] && [ -d "$_llamacpp_root/models" ]; then
    _lc_mc="$(/usr/bin/find "$_llamacpp_root/models" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)"
    _mc=$((_mc + _lc_mc))
fi

_is_uint() {
    case "$1" in
        ""|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

_gpu_q=""
_drv="unknown"
_cuda="unknown"
_gpu_ok=0
if command -v nvidia-smi >/dev/null 2>&1; then
    if _gpu_q_out="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,name --format=csv,noheader,nounits 2>/dev/null)"; then
        _gpu_q="$(printf '%s\n' "$_gpu_q_out" | sed -n '1p')"
        _drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p')"
        _cuda="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' | sed -n '1p')"
        _gpu_ok=1
    fi
fi

_gpu_util=0 _gpu_mem_used=0 _gpu_mem_total=0 _gpu_name="GPU unavailable"
if [ -n "$_gpu_q" ]; then
    _gpu_util="$(echo "$_gpu_q" | cut -d',' -f1 | tr -d ' ')"
    _gpu_mem_used="$(echo "$_gpu_q" | cut -d',' -f2 | tr -d ' ')"
    _gpu_mem_total="$(echo "$_gpu_q" | cut -d',' -f3 | tr -d ' ')"
    _gpu_name="$(echo "$_gpu_q" | cut -d',' -f4 | sed 's/^ //; s/ *$//')"
fi

_is_uint "$_gpu_util" || _gpu_util=0
_is_uint "$_gpu_mem_used" || _gpu_mem_used=0
_is_uint "$_gpu_mem_total" || _gpu_mem_total=0

_gpu_mem_used_gb="$(awk -v mem="$_gpu_mem_used" 'BEGIN {printf "%.1f", mem/1024}')"
_gpu_mem_total_gb="$(awk -v mem="$_gpu_mem_total" 'BEGIN {printf "%.1f", mem/1024}')"
if [ "$_gpu_mem_total" -gt 0 ]; then
    _mem_pct=$((_gpu_mem_used * 100 / _gpu_mem_total))
else
    _mem_pct=0
fi

_mkbar() {
    local pct=$1 width=20 color=$2
    local filled=$((pct * width / 100))
    local empty=$((width - filled))
    local i=0
    local bar_f="" bar_e=""

    while [ "$i" -lt "$filled" ]; do
        bar_f="${bar_f}█"
        i=$((i + 1))
    done
    i=0
    while [ "$i" -lt "$empty" ]; do
        bar_e="${bar_e}░"
        i=$((i + 1))
    done

    printf '%s%s' "$(_c "$color" "$bar_f")" "$(_c 238 "$bar_e")"
}

_proc_data=""
if [ "$_gpu_ok" -eq 1 ]; then
    if _proc_query="$(nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader,nounits 2>/dev/null)"; then
        _proc_data="$_proc_query"
    fi
fi

# ── Layout ───────────────────────────────────────────

_section_title() {
    local title="$1" color="$2"
    printf '%s %s %s\n' "$(_c 238 '╭─')" "$(_cb "$color" "$title")" "$(_c 238 '────────────────────────────────────────')"
}

_metric() {
    local label="$1" value="$2"
    printf '%s %s\n' "$(_c 244 "$(printf '%-13s' "$label")")" "$value"
}

_pill() {
    local color="$1" label="$2"
    printf '%s %s' "$(_c "$color" '●')" "$(_cb "$color" "$label")"
}

_muted_rule="$(_c 238 '╰───────────────────────────────────────────────')"

if [ "$_s" = "active" ]; then
    _status="$(_pill 78 'ONLINE')"
else
    _status="$(_pill 167 'OFFLINE')"
fi
_backend_label="${_active_backend:-standby}"
_endpoint="http://localhost:${_port}/v1"

_header="$(printf '%s %s %s %s\n%s\n' \
    "$(_cb 37 'vserve')" \
    "$(_c 245 'CONTROL DECK')" \
    "$(_c 238 '•')" \
    "$_status" \
    "$(_c 244 "backend ${_backend_label}  ·  ${_endpoint}")")"

_sec_server="$(printf '%s\n%s\n%s\n%s\n%s\n%s' \
    "$(_section_title 'SERVING' 78)" \
    "$(_metric STATE "$_status")" \
    "$(_metric MODEL "$(_c 252 "${_m:-no model active}")")" \
    "$(_metric ENDPOINT "$(_c 252 "$_endpoint")")" \
    "$(_metric CATALOG "$(_c 252 "${_mc} downloaded models")")" \
    "$_muted_rule")"

_gpu_state="$(_pill 78 'READY')"
if [ "$_gpu_ok" -ne 1 ]; then
    _gpu_state="$(_pill 220 'UNAVAILABLE')"
fi

_sec_gpu="$(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s' \
    "$(_section_title 'GPU TELEMETRY' 81)" \
    "$(_metric STATE "$_gpu_state")" \
    "$(_metric DEVICE "$(_c 252 "${_gpu_name}")")" \
    "$(_metric RUNTIME "$(_c 252 "${_drv}") $(_c 240 "CUDA ${_cuda}")")" \
    "$(_metric COMPUTE "$(_mkbar "$_gpu_util" 186)  $(_c 252 "${_gpu_util}%")")" \
    "$(_metric VRAM "$(_mkbar "$_mem_pct" 37)  $(_c 252 "${_gpu_mem_used_gb} / ${_gpu_mem_total_gb} GB")")" \
    "$_muted_rule")"

if [ -n "$_proc_data" ]; then
    _ph="$(printf '%s\n%s %s %s %s %s' \
        "$(_section_title 'GPU WORKLOAD' 186)" \
        "$(_cb 244 "$(printf '%-18s' PROCESS)")" \
        "$(_cb 244 "$(printf '%-12s' USER)")" \
        "$(_cb 244 "$(printf '%-12s' VRAM)")" \
        "$(_cb 244 "$(printf '%-12s' UPTIME)")" \
        "$(_cb 244 PID)")"

    _prows="$_ph"
    while IFS=',' read -r pid mem name; do
        pid="$(echo "$pid" | tr -d ' ')"
        mem="$(echo "$mem" | tr -d ' ')"
        name="$(echo "$name" | sed 's/^ //; s|.*/||')"
        _is_uint "$pid" || continue
        _is_uint "$mem" || mem=0
        mem_gb="$(awk -v mem="$mem" 'BEGIN {printf "%.1f GB", mem/1024}')"
        owner="$(ps -o user= -p "$pid" 2>/dev/null || echo "?")"
        uptime="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')" ; [ -z "$uptime" ] && uptime="?"
        _row="$(printf '%s %s %s %s %s\n' \
            "$(_c 252 "$(printf '%-18s' "$name")")" \
            "$(_c 240 "$(printf '%-12s' "$owner")")" \
            "$(_c 186 "$(printf '%-12s' "$mem_gb")")" \
            "$(_c 240 "$(printf '%-12s' "$uptime")")" \
            "$(_c 240 "$pid")")"
        _prows="$(printf '%s\n%s' "$_prows" "$_row")"
    done <<< "$_proc_data"
    _sec_proc="$(printf '%s\n%s' "$_prows" "$_muted_rule")"
else
    _sec_proc="$(printf '%s\n%s\n%s' \
        "$(_section_title 'GPU WORKLOAD' 186)" \
        "$(_c 240 'No GPU processes')" \
        "$_muted_rule")"
fi

_sep="$(_c 238 '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')"

_cmd_row() {
    printf '%s %s %s\n' "$(_c 238 '›')" "$(_c 37 "$(printf '%-28s' "$1")")" "$(_c 240 "$2")"
}

_sec_cmds="$(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s' \
    "$(_section_title 'COMMAND DECK' 37)" \
    "$(_cmd_row 'vserve list' 'list models with limits & capabilities')" \
    "$(_cmd_row 'vserve add [model]' 'search & download from HuggingFace')" \
    "$(_cmd_row 'vserve rm [model]' 'remove a downloaded model')" \
    "$(_cmd_row 'vserve tune [model]' 'calculate context & concurrency limits')" \
    "$(_cmd_row 'vserve run [model]' 'start serving (interactive config)')" \
    "$(_cmd_row 'vserve stop' 'stop the inference server')" \
    "$(_cmd_row 'vserve status' 'show what is currently serving')" \
    "$(_cmd_row 'vserve fan [auto|off|30-100]' 'GPU fan control')" \
    "$(_cmd_row 'vserve doctor' 'check system readiness')" \
    "$_muted_rule")"

# ── Update check ────────────────────────────────────
_update_cache="$HOME/.cache/vserve/update-check.json"
_sec_update=""
if [ -f "$_update_cache" ]; then
    _uc_current="$(grep -o '"current" *: *"[^"]*"' "$_update_cache" | sed 's/.*: *"//;s/"//')"
    _uc_latest="$(grep -o '"latest" *: *"[^"]*"' "$_update_cache" | sed 's/.*: *"//;s/"//')"
    _uc_has_update="$(grep -o '"update_available" *: *[^,}]*' "$_update_cache" | sed 's/.*: *//')"
    if [ "$_uc_has_update" = "true" ] && [ -n "$_uc_current" ] && [ -n "$_uc_latest" ]; then
        _sec_update="
$(_cb 220 '  Update available:') $(_c 252 " $_uc_current →") $(_cb 78 " $_uc_latest")
$(_c 243 '  Run: vserve update')"
    fi
fi

_body="
$_header

$_sec_server

$_sep

$_sec_gpu

$_sec_proc

$_sep

$_sec_cmds$_sec_update
"

gum style --border rounded --border-foreground 24 --padding "0 3" --margin "1 2" "$_body"

unset -f _c _cb _section_title _metric _pill _cmd_row _mkbar _is_uint
unset _esc _reset _vserve_cfg _cfg_values _vllm_root _llamacpp_root _llamacpp_svc _svc_name _port
unset _s _m _mc _lc_mc _active _active_backend _gpu_q _gpu_q_out _gpu_ok _drv _cuda _gpu_util _gpu_mem_used _gpu_mem_total _gpu_name
unset _gpu_mem_used_gb _gpu_mem_total_gb _mem_pct
unset _proc_data _proc_query _ph _prows _row
unset _update_cache _uc_current _uc_latest _uc_has_update
unset _header _status _backend_label _endpoint _gpu_state _muted_rule _sec_server _sec_gpu _sec_proc _sec_cmds _sep _sec_update _body
