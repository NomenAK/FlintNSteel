# Run a .bat file, write its output to a log, and print the last 20 lines and its exit code.
param([Parameter(Mandatory)][string]$Bat, [Parameter(Mandatory)][string]$Log)
& cmd.exe /c "`"$Bat`"" *> $Log
$code = $LASTEXITCODE
Get-Content $Log -Tail 20
"exit=$code"
exit $code
