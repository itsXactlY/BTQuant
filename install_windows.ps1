Set-StrictMode -Version Latest

function Is-SdkInstalled {
    try {
        $sdkPath = (Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots" -ErrorAction Stop).KitsRoot10
        $sdkVersions = Get-ChildItem -Path "$sdkPath\bin" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
        if ($sdkVersions -match "10\.0\.(\d+)\.\d+") {
            Write-Host "Windows SDK detected: $sdkVersions"
            return $true
        }
    }
    catch {
        Write-Host "Windows SDK not found in registry."
    }
    return $false
}

function Install-BuildTools {
    Write-Host "Installing Microsoft Visual C++ Build Tools and Windows SDK..."
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
            "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041", `
            "--add", "Microsoft.VisualStudio.Component.Windows11SDK.22000" `
            -NoNewWindow -Wait -PassThru

        if ($process.ExitCode -eq 0 -or $process.ExitCode -eq 3010) {
            Write-Host "Visual Studio Build Tools and Windows SDK installed successfully."
            return $true
        }
        Write-Error "Installation failed with exit code: $($process.ExitCode)"
        return $false
    }
    catch {
        Write-Error "Failed to install dependencies: $_"
        return $false
    }
    finally {
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
    }
}

function Install-ODBC {
    Write-Host "Checking for ODBC Driver..."
    $odbcRegistryPath = "HKLM:\SOFTWARE\ODBC\ODBCINST.INI\ODBC Driver 18 for SQL Server"
    
    if (Test-Path $odbcRegistryPath) {
        Write-Host "ODBC Driver 18 for SQL Server is already installed."
        return $true
    }

    Write-Host "Downloading and Installing Microsoft ODBC Driver for SQL Server..."
    try {
        $odbcDownloadUrl = "https://go.microsoft.com/fwlink/?linkid=2280794"
        $installerPath = "$env:TEMP\msodbcsql18.msi"

        Invoke-WebRequest -Uri $odbcDownloadUrl -OutFile $installerPath

        Start-Process -FilePath "msiexec.exe" -ArgumentList "/i $installerPath /quiet /norestart" -NoNewWindow -Wait

        Remove-Item $installerPath -Force
        Write-Host "ODBC Driver installed successfully."
        return $true
    }
    catch {
        Write-Error "Failed to install ODBC Driver: $_"
        return $false
    }
}

Write-Host "Starting BTQuant installation script..."

Write-Host "Checking for Visual C++ Build Tools..."
if (-not (Test-Path "C:\BuildTools\VC\Auxiliary\Build\vcvarsall.bat")) {
    Write-Host "Visual C++ Build Tools not found."
    if (-not (Install-BuildTools)) {
        Write-Error "Visual C++ Build Tools installation failed. Aborting."
        exit 1
    }
}

Write-Host "Checking for Windows SDK..."
if (-not (Is-SdkInstalled)) {
    Write-Host "Windows SDK not found. Installing..."
    if (-not (Install-BuildTools)) {
        Write-Error "Windows SDK installation failed. Aborting."
        exit 1
    }
}

Write-Host "Checking for ODBC Driver..."
$odbcInstalled = Get-ItemProperty -Path "HKLM:\SOFTWARE\ODBC\ODBCINST.INI\ODBC Driver 18 for SQL Server" -ErrorAction SilentlyContinue
if (-not $odbcInstalled) {
    Install-ODBC
} else {
    Write-Host "ODBC Driver is already installed."
}

Write-Host "All dependencies verified. Proceeding with BTQuant installation..."

if (Test-Path "BTQuant") {
    Write-Host "Removing previous BTQuant installation..."
    Remove-Item -Recurse -Force "BTQuant"
}

Write-Host "Cloning BTQuant repository..."
git clone https://github.com/itsXactlY/BTQuant
Set-Location BTQuant

$repoRoot = Get-Location
Write-Host "Repository root: $repoRoot"

Write-Host "Setting up virtual environment..."
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel

$venvSitePackages = Join-Path $repoRoot ".venv\Lib\site-packages"
Write-Host "Using virtual environment site-packages at: $venvSitePackages"

Write-Host "Installing dependencies..."

if (Test-Path "dependencies") {
    Set-Location "dependencies"
} else {
    Write-Error "Dependencies folder not found."
    exit 1
}

Write-Host "Installing pybind11..."
python -m pip install --target $venvSitePackages pybind11

$localPackages = @(".", "fastquant", "BTQ_Exchanges", "MsSQL")
foreach ($package in $localPackages) {
    Write-Host "Installing $package..."
    $packagePath = Join-Path (Get-Location) $package
    if (Test-Path $packagePath) {
        Push-Location $packagePath
        python -m pip install --target $venvSitePackages .
        Pop-Location
    }
    else {
        Write-Host "Warning: Package folder for $package not found. Skipping..."
    }
}

Write-Host "Checking for fast_mssql.pyd..."
$pydFile = Get-ChildItem -Recurse -Filter "fast_mssql*.pyd" | Select-Object -First 1
if ($pydFile) {
    $targetDir = Join-Path $venvSitePackages "fastquant\data\mssql\MsSQL"
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Copy-Item $pydFile.FullName -Destination $targetDir -Force
}

Set-Location $repoRoot

Write-Host "`nBTQuant installation complete!`n"
Write-Host "To use BTQuant in a new PowerShell session:"
Write-Host "1. cd BTQuant"
Write-Host "2. .\.venv\Scripts\Activate.ps1"
Write-Host "3. Start using BTQuant with Python"

Read-Host -Prompt "Press Enter to exit"
