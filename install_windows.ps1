# Set strict mode for better error handling
Set-StrictMode -Version Latest

# Function to install Visual C++ Build Tools
function Install-BuildTools {
    Write-Host "Installing Microsoft Visual C++ Build Tools..."
    $buildToolsUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
    $installerPath = "$env:TEMP\vs_buildtools.exe"
    
    try {
        Write-Host "Downloading Visual Studio Build Tools..."
        Invoke-WebRequest -Uri $buildToolsUrl -OutFile $installerPath
        
        Write-Host "Installing Visual Studio Build Tools (this may take a while)..."
        $process = Start-Process -FilePath $installerPath -ArgumentList `
            "--quiet", "--wait", "--norestart", "--nocache", `
            "--installPath", "C:\BuildTools", `
            "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", `
            "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041" `
            -NoNewWindow -Wait -PassThru

        if ($process.ExitCode -eq 0 -or $process.ExitCode -eq 3010) {
            Write-Host "Visual Studio Build Tools installed successfully."
            return $true
        }
        Write-Error "Visual Studio Build Tools installation failed with exit code: $($process.ExitCode)"
        return $false
    }
    catch {
        Write-Error "Failed to install Visual Studio Build Tools: $_"
        return $false
    }
    finally {
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
    }
}

# Main installation script
Write-Host "Starting BTQuant installation script..."

# Install Visual C++ Build Tools if needed
Write-Host "Checking for Visual C++ Build Tools..."
if (-not (Test-Path "C:\BuildTools\VC\Auxiliary\Build\vcvarsall.bat")) {
    Write-Host "Visual C++ Build Tools not found."
    if (-not (Install-BuildTools)) {
        Write-Error "Visual C++ Build Tools installation failed. Aborting."
        exit 1
    }
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# Clean up any existing BTQuant directory
if (Test-Path "BTQuant") {
    Remove-Item -Recurse -Force "BTQuant"
}

# Clone repository
Write-Host "Cloning BTQuant repository..."
git clone https://github.com/itsXactlY/BTQuant
Set-Location BTQuant

# Create and activate virtual environment
Write-Host "Setting up virtual environment..."
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"

# Upgrade base packages
Write-Host "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "Installing dependencies..."
Set-Location dependencies

$dependencies = @(
    ".\BTQ_Exchanges",
    "pybind11",
    ".\MsSQL",
    "backtrader",
    ".\fastquant"
)

foreach ($dep in $dependencies) {
    Write-Host "Installing $dep..."
    python -m pip install --no-cache-dir $dep
}

# Handle fast_mssql.pyd file
Write-Host "Checking for fast_mssql.pyd..."
$pydFile = Get-ChildItem -Recurse -Filter "fast_mssql*.pyd" | Select-Object -First 1
if ($pydFile) {
    $targetDir = ".\.venv\Lib\site-packages\fastquant\data\mssql\MsSQL"
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Copy-Item $pydFile.FullName -Destination $targetDir -Force
}

Write-Host "`nBTQuant installation complete!`n"
Write-Host "To use BTQuant in a new PowerShell session:"
Write-Host "1. cd BTQuant"
Write-Host "2. .\.venv\Scripts\Activate.ps1"
Write-Host "3. Start using BTQuant with Python"

Read-Host -Prompt "Press Enter to exit"