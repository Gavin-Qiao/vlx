#!/bin/bash
# vserve login banner — reads from vserve config (single source of truth)
# Installed by: vserve init → ~/.config/vserve/welcome.sh
# Remove with:  rm ~/.config/vserve/welcome.sh  (and delete the source line from your shell rc)

[ -z "$BASH_VERSION" ] && [ -z "$ZSH_VERSION" ] && return 0
command -v gum &>/dev/null || return 0

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
    "$(gum style --bold --foreground 37 'vserve')" \
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
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve list')"              "$(gum style --foreground 240 'list models with limits & capabilities')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve add [model]')"       "$(gum style --foreground 240 'search & download from HuggingFace')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve rm [model]')"        "$(gum style --foreground 240 'remove a downloaded model')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve tune [model]')"      "$(gum style --foreground 240 'calculate context & concurrency limits')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve run [model]')"       "$(gum style --foreground 240 'start serving (interactive config)')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve stop')"              "$(gum style --foreground 240 'stop the inference server')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve status')"            "$(gum style --foreground 240 'show what is currently serving')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve fan [auto|off|30-100]')" "$(gum style --foreground 240 'GPU fan control')")" \
    "$(gum join --horizontal "$(gum style --foreground 37 --width 28 'vserve doctor')"            "$(gum style --foreground 240 'check system readiness')")" \
)"

# ── Update check ────────────────────────────────────
_update_cache="$HOME/.cache/vserve/update-check.json"
_sec_update=""
if [ -f "$_update_cache" ]; then
    _uc_current="$(grep -o '"current" *: *"[^"]*"' "$_update_cache" | sed 's/.*: *"//;s/"//')"
    _uc_latest="$(grep -o '"latest" *: *"[^"]*"' "$_update_cache" | sed 's/.*: *"//;s/"//')"
    _uc_has_update="$(grep -o '"update_available" *: *[^,}]*' "$_update_cache" | sed 's/.*: *//')"
    if [ "$_uc_has_update" = "true" ] && [ -n "$_uc_current" ] && [ -n "$_uc_latest" ]; then
        _sec_update="$(gum join --vertical \
            "" \
            "$(gum join --horizontal \
                "$(gum style --foreground 220 --bold "  Update available:")" \
                "$(gum style --foreground 252 " $_uc_current →")" \
                "$(gum style --foreground 78 --bold " $_uc_latest")")" \
            "$(gum style --foreground 243 "  Run: vserve update")")"
    fi
fi

_body="$(gum join --vertical \
    "" "$_header" "" "$_sec_server" "" "$_sep" "" "$_sec_gpu" "" \
    "$_sec_proc" "" "$_sep" "" "$_sec_cmds" "$_sec_update" "" \
)"

gum style --border rounded --border-foreground 24 --padding "0 3" --margin "1 2" "$_body"

unset -f _lbl _mkbar
unset _vserve_cfg _cfg_values _vllm_root _llamacpp_root _llamacpp_svc _svc_name _port
unset _s _m _mc _lc_mc _active _active_backend _gpu_q _drv _cuda _gpu_util _gpu_mem_used _gpu_mem_total _gpu_name
unset _gpu_mem_used_gb _gpu_mem_total_gb _mem_pct
unset _proc_data _ph _prows _row
unset _update_cache _uc_current _uc_latest _uc_has_update
unset _header _status _sec_server _sec_gpu _sec_proc _sec_cmds _sep _sec_update _body
