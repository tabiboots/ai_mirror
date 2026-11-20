param(
    [int]$ProcId,
    [string]$AvatarDir = "vendor\LiveTalking\data\avatars\tabi_musetalk",
    [int]$StallSeconds = 240,
    [int]$IntervalSeconds = 15,
    [string]$Log = "results\avatar_watch.log"
)

New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
$lastCpu = -1
$lastChange = Get-Date

function Snapshot {
    $files = @("coords.pkl", "mask_coords.pkl", "latents.pt") | ForEach-Object { if (Test-Path "$AvatarDir\$_") { $_ } }
    $masks = (Get-ChildItem "$AvatarDir\mask" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    "masks=$masks outputs=[$($files -join ',')] gpu=[$gpu]"
}

while ($true) {
    $p = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
    $snap = Snapshot
    if (-not $p) {
        $line = "{0} EXITED {1}" -f (Get-Date -Format HH:mm:ss), $snap
        $line | Tee-Object -FilePath $Log -Append
        break
    }
    $cpu = [math]::Round($p.CPU)
    if ($cpu -ne $lastCpu -or $snap -ne $script:lastSnap) { $lastChange = Get-Date }
    $lastCpu = $cpu
    $script:lastSnap = $snap
    $idle = [int]((Get-Date) - $lastChange).TotalSeconds
    "{0} running cpu={1}s idle={2}s {3}" -f (Get-Date -Format HH:mm:ss), $cpu, $idle, $snap | Add-Content $Log
    if ($idle -ge $StallSeconds) {
        "{0} STALLED: no CPU or output change for {1}s. {2}" -f (Get-Date -Format HH:mm:ss), $idle, $snap | Tee-Object -FilePath $Log -Append
        break
    }
    Start-Sleep $IntervalSeconds
}
