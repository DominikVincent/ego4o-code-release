# Sourced by the stage scripts. check_gpus_free "2,3" aborts if any of the
# requested GPUs already has >2 GiB allocated (other users share this server).
check_gpus_free() {
    local ids="${1//,/ }"
    for id in $ids; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$id")
        if [ "$used" -gt 2048 ]; then
            echo "ABORT: GPU $id is in use (${used} MiB allocated). Pick free GPUs via GPUS=..." >&2
            exit 1
        fi
    done
    echo "GPUs $1 are free."
}
