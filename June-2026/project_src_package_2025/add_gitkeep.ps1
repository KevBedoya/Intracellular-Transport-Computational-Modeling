# PowerShell Script to create .gitkeep in all subdirectories under data_output
$root = "data_output"

if (-Not (Test-Path $root)) {
    Write-Host "Directory 'data_output' not found."
    exit
}

# Recursively find all directories and add .gitkeep
Get-ChildItem -Path $root -Recurse -Directory | ForEach-Object {
    $gitkeepPath = Join-Path $_.FullName ".gitkeep"
    if (-Not (Test-Path $gitkeepPath)) {
        New-Item -Path $gitkeepPath -ItemType File | Out-Null
        Write-Host "Created .gitkeep in $($_.FullName)"
    }
}

# run using:
# powershell -ExecutionPolicy Bypass -File add_gitkeep.ps1