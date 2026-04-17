# BTQuant Installation Guide

## Supported Platforms

- Ubuntu / Debian
- Fedora
- CentOS / RHEL
- Arch Linux / Manjaro / EndeavourOS / Garuda

The installer script auto-detects your distribution. Other Linux distributions are not supported.

## Prerequisites

- Linux operating system
- sudo/root access
- Internet connection
- Git

## Automated Installation

The `install_all.sh` script handles the complete setup:

```bash
git clone --recurse-submodules -b prototyping https://github.com/ItsXactlY/BTQuant BTQuant
cd BTQuant
bash Installers/install_all.sh
```

### What the Installer Does

The script runs these steps in order:

1. **Detects your Linux distribution** from `/etc/os-release`

2. **Installs system dependencies** based on your distro:

   Ubuntu/Debian:
   ```
   build-essential, python3-dev, python3-venv, python3-pybind11,
   unixodbc-dev, git, cmake, libssl-dev, libboost-all-dev,
   rapidjson-dev, curl, wget, gnupg2, software-properties-common, lsb-release
   ```

   Fedora:
   ```
   Development Tools group, python3-devel, python3-pybind11,
   unixODBC-devel, git, cmake, openssl-devel, boost-devel,
   rapidjson-devel, curl, wget, gnupg2
   ```

   CentOS/RHEL:
   ```
   Development Tools group, python3-devel, python3-pybind11,
   unixODBC-devel, git, cmake, openssl-devel, boost-devel,
   rapidjson-devel, curl, wget, gnupg2
   ```

   Arch-based:
   ```
   base-devel, python, pybind11, unixodbc, git, cmake,
   openssl, boost, rapidjson, curl, wget, gnupg
   ```

3. **Installs Microsoft SQL Server** (skipped if already running):
   - Adds the Microsoft package repository for your distro
   - Installs `mssql-server` and `msodbcsql18` ODBC driver
   - Runs `mssql-conf setup` with an evaluation license
   - Enables and starts the `mssql-server` systemd service
   - Waits 30 seconds for SQL Server to start

4. **Installs sqlcmd tools** (skipped if already available):
   - Installs `mssql-tools18` (or `mssql-tools` on Arch)
   - Adds to PATH via `~/.bashrc`

5. **Sets up BTQuant Python environment**:
   - Clones the repository (prototyping branch) if not present
   - Creates a virtual environment at `~/.btq`
   - Upgrades pip, setuptools, wheel
   - Installs pybind11 in the venv
   - Runs `pip install .` from the `dependencies/` directory (installs all Python dependencies: ccxt, pybind11, pyodbc, websockets, Web3, matplotlib, numpy, polars, pyarrow, telethon, scikit-learn, keras, pytz, optuna)

6. **Builds the Fast_MSSQL driver**:
   - Checks for a pre-compiled `.so` binary matching your Python version
   - If found, copies it to the venv site-packages
   - If not found, builds from source via `pip install .` in `dependencies/MsSQL/`

7. **Initializes the database**:
   - Checks if `BTQ_MarketData` database exists on SQL Server
   - If not, runs `dependencies/datacollector/init_database.py`

8. **Builds CCAPI**:
   - Initializes git submodules (`git submodule update --init --recursive`)
   - Runs bootstrap script if present
   - Builds with CMake
   - Copies `market_data_collector` and `hotspine` binaries to `~/bin/`

### Post-Installation

After the script completes:

```bash
# Activate the virtual environment
source ~/.btq/bin/activate

# Reload PATH for sqlcmd and ~/bin
source ~/.bashrc
```

The installer prints the SQL Server SA password. Save it if you need SQL Server access.

## Manual Installation

If the automated installer does not work for your setup:

### Step 1: Clone the Repository

```bash
git clone --recurse-submodules -b prototyping https://github.com/ItsXactlY/BTQuant BTQuant
cd BTQuant
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv ~/.btq
source ~/.btq/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install pybind11
cd dependencies
pip install .
```

This installs all required packages from `dependencies/setup.py`:
- ccxt, pybind11, pyodbc, websockets, websocket-client, Web3
- matplotlib, numpy, polars, pyarrow
- telethon, scikit-learn, keras, pytz, optuna

### Step 4: Build Fast_MSSQL (Optional)

Only needed if you use SQL Server data feeds:

```bash
cd dependencies/MsSQL
pip install .
```

Or copy a pre-compiled binary from `dependencies/backtrader/feeds/mssql/` to your site-packages.

### Step 5: Build CCAPI (Optional)

Only needed for real-time market data collection:

```bash
cd dependencies/ccapi
git submodule update --init --recursive
cd example
mkdir -p build && cd build
cmake ..
cmake --build .
```

## Configuration

### Credentials and Secrets

Edit `dependencies/backtrader/dontcommit.py` to configure:

```python
# JackRabbitRelay
identify = ""                    # JRR identify string
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = "/home/JackrabbitRelay2/Data/Mimic/"

# Web3 (BSC)
bsc_privaccount1 = ""
bsc_privaccountaddress = ""

# Solana
solana_privkey_base58 = ""
solana_wallet_address = ""

# Discord
discord_webhook_url = ""

# Telegram
telegram_api_id = 1111111
telegram_api_hash = ""
telegram_session_file = ".base.session"
telegram_channel = -100

# SQL Server (match your MSSQL setup)
server = 'localhost'
candle_database = 'BinanceData'
optuna_database = 'OptunaBT'
username = 'SA'
password = 'YourStrong!Passw0rd'
```

The `dontcommit.py` file is listed in `.gitignore` -- do not commit credentials.

### SQL Server Connection Strings

Two connection strings are auto-generated from the settings above:
- `connection_string` -- for the candle/market data database (`candle_database`)
- `optuna_connection_string` -- for the Optuna optimization database (`optuna_database`)

Both use ODBC Driver 18 with `TrustServerCertificate=yes`.

### Data Cache

BTQuant caches market data as Parquet files in `.btq_cache/` (or the path set by `BTQ_CACHE_DIR` environment variable). Clear the cache with:

```bash
btq --clear-cache backtest --coin BTC --strategy MyStrategy
```

## Verifying the Installation

```bash
# Activate the environment
source ~/.btq/bin/activate

# Check CLI is available
btq --help

# List available strategies
btq list strategies

# Run a simple backtest (requires data source)
btq backtest --coin BTC --strategy VuManchCipher_A --interval 15m --start 2024-01-01 --end 2024-01-08
```

## Troubleshooting

### Python version mismatch

BTQuant targets Python 3.13. Verify with:
```bash
source ~/.btq/bin/activate
python --version
```

### SQL Server not starting

```bash
sudo systemctl status mssql-server
sudo journalctl -u mssql-server -f
```

### Missing system libraries

Re-run the system dependency installation for your distro (see step 2 of the installer).

### Fast_MSSQL binary not found

If the pre-compiled `.so` is not available for your Python version, build from source:
```bash
cd dependencies/MsSQL
pip install .
```

### CCAPI build failures

Ensure you have CMake 3.15+ and a C++17 compiler:
```bash
g++ --version
cmake --version
```

### Import errors

Make sure you are in the virtual environment and installed from the `dependencies/` directory:
```bash
source ~/.btq/bin/activate
cd dependencies
pip install .
```

## Uninstallation

```bash
# Remove the virtual environment
rm -rf ~/.btq

# Remove the repository
rm -rf BTQuant

# Optionally remove SQL Server (Ubuntu/Debian)
sudo apt-get remove -y mssql-server msodbcsql18 mssql-tools18
```