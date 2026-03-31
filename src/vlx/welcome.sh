#!/bin/bash
# vlx login banner — reads from vlx config (single source of truth)
# Installed by: vlx init → ~/.config/vx/welcome.sh
# Remove with:  rm ~/.config/vx/welcome.sh  (and delete the source line from your shell rc)

[ -z "$BASH_VERSION" ] && [ -z "$ZSH_VERSION" ] && return 0
command -v gum &>/dev/null || return 0

# ── Read vlx config ──────────────────────────────────

_vlx_cfg="$HOME/.config/vx/config.yaml"
_vllm_root="/opt/vllm"
_svc_name="vllm"
_port="8888"

if [ -f "$_vlx_cfg" ]; then
    _vllm_root="$(grep '^vllm_root:' "$_vlx_cfg" 2>/dev/null | sed 's/vllm_root: *//')" || _vllm_root="/opt/vllm"
    _svc_name="$(grep '^service_name:' "$_vlx_cfg" 2>/dev/null | sed 's/service_name: *//')" || _svc_name="vllm"
    _port="$(grep '^port:' "$_vlx_cfg" 2>/dev/null | sed 's/port: *//')" || _port="8888"
fi

# ── Gather data ──────────────────────────────────────

_s="inactive"
systemctl is-active --quiet "$_svc_name" 2>/dev/null && _s="active"

_m=""
_active="$_vllm_root/configs/active.yaml"
if [ -f "$_active" ]; then
    _m="$(grep '^model:' "$_active" 2>/dev/null | sed "s|model: *${_vllm_root}/models/||; s|model: *||")"
fi

_mc=0
if [ -d "$_vllm_root/models" ]; then
    _mc="$(/usr/bin/find "$_vllm_root/models" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)"
fi

_gpu_q="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,name --format=csv,noheader,nounits 2>/dev/null)"
_drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"
_cuda="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+')"

_gpu_util=0 _gpu_mem_used=0 _gpu_mem_total=1 _gpu_name="unknown"
if [ -n "$_gpu_q" ]; then
    _gpu_util="$(echo "$_gpu_q" | cut -d',' -f1 | tr -d ' ')"
    _gpu_mem_used="$(echo "$_gpu_q" | cut -d',' -f2 | tr -d ' ')"
    _gpu_mem_total="$(echo "$_gpu_q" | cut -d',' -f3 | tr -d ' ')"
    _gpu_name="$(echo "$_gpu_q" | cut -d',' -f4 | sed 's/^ //; s/ *$//')"
fi

_gpu_mem_used_gb="$(awk "BEGIN {printf \"%.1f\", ${_gpu_mem_used}/1024}")"
_gpu_mem_total_gb="$(awk "BEGIN {printf \"%.1f\", ${_gpu_mem_total}/1024}")"
_mem_pct=$((_gpu_mem_used * 100 / _gpu_mem_total))

_mkbar() {
    local pct=$1 width=20 color=$2
    local filled=$((pct * width / 100))
    local empty=$((width - filled))
    local bar_f="" bar_e=""
    [ $filled -gt 0 ] && bar_f="$(printf '%0.s█' $(seq 1 $filled))"
    [ $empty -gt 0 ]  && bar_e="$(printf '%0.s░' $(seq 1 $empty))"
    echo "$(gum style --foreground "$color" "$bar_f")$(gum style --foreground 238 "$bar_e")"
}

_proc_data="$(nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader,nounits 2>/dev/null)"

# ── Layout ───────────────────────────────────────────

_lbl() { gum style --foreground 243 --width 16 "$1"; }

[ "$_s" = "active" ] \
    && _status="$(gum style --foreground 78 --bold 'ONLINE')" \
    || _status="$(gum style --foreground 167 'OFFLINE')"

_header="$(gum join --horizontal \
    "$(gum style --bold --foreground 37 'vlx')" \
    "$(gum style --foreground 243 '  inference server')")"

_sec_server="$(gum join --vertical \
    "$(gum join --horizontal "$(_lbl STATUS)"   "$_status")" \
    "$(gum join --horizontal "$(_lbl MODEL)"    "$(gum style --foreground 252 "${_m:-no model active}")")" \
    "$(gum join --horizontal "$(_lbl ENDPOINT)" "$(gum style --foreground 252 "http://localhost:${_port}/v1")")" \
    "$(gum join --horizontal "$(_lbl MODELS)"   "$(gum style --foreground 252 "${_mc} downloaded")")" \
)"

_sec_gpu="$(gum join --vertical \
    "$(gum join --horizontal "$(_lbl DEVICE)"  "$(gum style --foreground 252 "${_gpu_name}")")" \
    "$(gum join --horizontal "$(_lbl DRIVER)"  "$(gum style --foreground 252 "${_drv}")" "$(gum style --foreground 240 "  CUDA ${_cuda}")")" \
    "$(gum join --horizontal "$(_lbl COMPUTE)" "$(_mkbar "$_gpu_util" 186)" "  $(gum style --foreground 252 "${_gpu_util}%")")" \
    "$(gum join --horizontal "$(_lbl VRAM)"    "$(_mkbar "$_mem_pct" 37)"   "  $(gum style --foreground 252 "${_gpu_mem_used_gb} / ${_gpu_mem_total_gb} GB")")" \
)"

if [ -n "$_proc_data" ]; then
    _ph="$(gum join --horizontal \
        "$(gum style --foreground 243 --bold --width 18 'PROCESS')" \
        "$(gum style --foreground 243 --bold --width 12 'USER')" \
        "$(gum style --foreground 243 --bold --width 12 'VRAM')" \
        "$(gum style --foreground 243 --bold --width 12 'UPTIME')" \
        "$(gum style --foreground 243 --bold 'PID')")"

    _prows="$_ph"
    while IFS=',' read -r pid mem name; do
        pid="$(echo "$pid" | tr -d ' ')"
        mem="$(echo "$mem" | tr -d ' ')"
        name="$(echo "$name" | sed 's/^ //; s|.*/||')"
        mem_gb="$(awk "BEGIN {printf \"%.1f GB\", ${mem}/1024}")"
        owner="$(ps -o user= -p "$pid" 2>/dev/null || echo "?")"
        uptime="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')" ; [ -z "$uptime" ] && uptime="?"
        _row="$(gum join --horizontal \
            "$(gum style --foreground 252 --width 18 "$name")" \
            "$(gum style --foreground 240 --width 12 "$owner")" \
            "$(gum style --foreground 186 --width 12 "$mem_gb")" \
            "$(gum style --foreground 240 --width 12 "$uptime")" \
            "$(gum style --foreground 240 "$pid")")"
        _prows="$(gum join --vertical "$_prows" "$_row")"
    done <<< "$_proc_data"
    _sec_proc="$_prows"
else
    _sec_proc="$(gum style --foreground 240 'No GPU processes')"
fi

_sep="$(gum style --foreground 238 '────────────────────────────────────────────────────────')"

_sec_cmds="$(gum join --vertical \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vlx download [model]')"    "$(gum style --foreground 240 'search & download from HuggingFace')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vlx start [model]')"        "$(gum style --foreground 240 'start serving (interactive config)')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vlx stop')"              "$(gum style --foreground 240 'stop the vLLM service')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vlx status')"            "$(gum style --foreground 240 'show what is currently serving')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vlx models')"            "$(gum style --foreground 240 'list models with limits & profiles')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vlx tune [model] [prof]')"  "$(gum style --foreground 240 'probe limits + benchmark')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vlx fan [auto|off|30-100]')" "$(gum style --foreground 240 'GPU fan control')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vlx doctor')"            "$(gum style --foreground 240 'check system readiness')")" \
)"

_body="$(gum join --vertical \
    "" "$_header" "" "$_sec_server" "" "$_sep" "" "$_sec_gpu" "" \
    "$_sec_proc" "" "$_sep" "" "$_sec_cmds" "" \
)"

gum style --border rounded --border-foreground 24 --padding "0 3" --margin "1 2" "$_body"

unset -f _lbl _mkbar
unset _vlx_cfg _vllm_root _svc_name _port
unset _s _m _mc _active _gpu_q _drv _cuda _gpu_util _gpu_mem_used _gpu_mem_total _gpu_name
unset _gpu_mem_used_gb _gpu_mem_total_gb _mem_pct
unset _proc_data _ph _prows _row
unset _header _status _sec_server _sec_gpu _sec_proc _sec_cmds _sep _body
