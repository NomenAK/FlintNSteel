# Machine-state report to run before a measurement. It describes the machine that runs it and writes nothing to the repository.
# Requires nvidia-smi on PATH. Do not commit its output.
Get-PSDrive -PSProvider FileSystem |
    Select-Object Name, @{n = 'FreeGB'; e = { [math]::Round($_.Free / 1e9, 1) } } |
    Format-Table -AutoSize
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,temperature.gpu,power.limit,display_active --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
(powercfg /getactivescheme)
