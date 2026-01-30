#!/bin/bash

# Microsoft SQL Server Installer for Arch Linux using AUR
# Installs SQL Server, creates database, and prepares for Binance Vision data

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
MSSQL_SA_PASSWORD=""
DATABASE_NAME="BinanceData"
USER_NAME="binance_user"
USER_PASSWORD=""
DATA_IMPORT_PATH=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Generate strong password
generate_strong_password() {
    local length=${1:-16}
    # Generate password with uppercase, lowercase, digits, and symbols
    python3 -c "
import random
import string

# Define character sets
upper = string.ascii_uppercase
lower = string.ascii_lowercase
digits = string.digits
symbols = '!@#$%^&*()_+-=[]{}|;:,.<>?'

# Ensure at least one character from each set
password = [
    random.choice(upper),
    random.choice(lower),  
    random.choice(digits),
    random.choice(symbols)
]

# Fill the rest randomly
all_chars = upper + lower + digits + symbols
for _ in range($length - 4):
    password.append(random.choice(all_chars))

# Shuffle and join
random.shuffle(password)
print(''.join(password))
"
}

# Detect Linux distribution and ensure it's Arch-based
detect_distro() {
    print_status "Detecting Linux distribution..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        case "$DISTRO" in
            arch|garuda|cachyos|manjaro|endeavouros)
                print_success "Detected Arch-based distribution: $DISTRO"
                ;;
            *)
                print_error "This installer is designed for Arch Linux and derivatives only."
                print_error "Detected: $DISTRO"
                exit 1
                ;;
        esac
    else
        print_error "Cannot detect distribution. This installer requires Arch Linux."
        exit 1
    fi
}

# Install AUR helper if not present
install_aur_helper() {
    if command_exists yay; then
        print_status "Found yay AUR helper"
        AUR_HELPER="yay"
    elif command_exists paru; then
        print_status "Found paru AUR helper"
        AUR_HELPER="paru"
    else
        print_status "No AUR helper found. Installing yay..."
        cd /tmp
        git clone https://aur.archlinux.org/yay.git
        cd yay
        makepkg -si --noconfirm
        cd ~
        AUR_HELPER="yay"
        print_success "yay installed successfully"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Update system
    sudo pacman -Syu --noconfirm
    
    # Install base dependencies from official repos
    sudo pacman -S --noconfirm --needed \
        base-devel \
        git \
        unixodbc \
        python \
        python-pip \
        python-pandas \
        python-sqlalchemy
    
    print_success "Official packages installed successfully"
    
    # Install AUR packages
    print_status "Installing AUR packages..."
    $AUR_HELPER -S --noconfirm python-pyodbc
        
    print_success "AUR packages installed successfully"
}

# Generate passwords and get user input
get_user_input() {
    print_status "Generating strong passwords..."
    
    # Generate strong SA password
    MSSQL_SA_PASSWORD=$(generate_strong_password 20)
    print_success "Strong SA password generated"
    
    # Generate strong user password
    USER_PASSWORD=$(generate_strong_password 16)
    print_success "Strong user password generated"
    
    echo
    read -p "Do you want to import Binance Vision data after setup? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -n "Enter path to Binance Vision data directory (e.g., /home/user/binance-data/spot/monthly/klines): "
        read DATA_IMPORT_PATH
        
        if [ ! -d "$DATA_IMPORT_PATH" ]; then
            print_warning "Directory doesn't exist yet, will create import script for later use"
            DATA_IMPORT_PATH=""
        fi
    fi
    
    print_success "Configuration set"
}

# Install SQL Server from AUR
install_sqlserver() {
    print_status "Installing Microsoft SQL Server from AUR..."
    
    # Install mssql-server
    $AUR_HELPER -S --noconfirm mssql-server
    
    print_success "SQL Server installed from AUR"
}

# Configure SQL Server with automated responses
configure_sqlserver() {
    print_status "Configuring SQL Server automatically..."
    
    # Create expect script to automate the configuration
    cat > /tmp/mssql_config.exp << 'EOF'
#!/usr/bin/expect -f
set timeout 60

# Get the password from environment
set sa_password $env(MSSQL_SA_PASSWORD)

spawn sudo env MSSQL_SA_PASSWORD=$sa_password ACCEPT_EULA=Y /opt/mssql/bin/mssql-conf setup

# Handle sudo password prompt
expect {
    "Passwort für*" {
        stty -echo
        send_user "Enter sudo password: "
        expect_user -re "(.*)\n"
        send_user "\n"
        stty echo
        send "$expect_out(1,string)\r"
        exp_continue
    }
    "Password for*" {
        stty -echo
        send_user "Enter sudo password: "
        expect_user -re "(.*)\n"
        send_user "\n"
        stty echo
        send "$expect_out(1,string)\r"
        exp_continue
    }
    "Edition eingeben*" {
        send "2\r"
        exp_continue
    }
    "Enter your edition*" {
        send "2\r"
        exp_continue
    }
    "Geben Sie Option*" {
        send "1\r"
        exp_continue
    }
    "Enter an option*" {
        send "1\r"
        exp_continue
    }
    "Das angegebene Kennwort*" {
        puts "Password complexity error, but continuing..."
        exp_continue
    }
    "password does not meet*" {
        puts "Password complexity error, but continuing..."
        exp_continue
    }
    eof
}
EOF

    # Make expect script executable and run it
    chmod +x /tmp/mssql_config.exp
    MSSQL_SA_PASSWORD="$MSSQL_SA_PASSWORD" /tmp/mssql_config.exp
    
    # Clean up expect script
    rm -f /tmp/mssql_config.exp
    
    # Enable and start SQL Server service
    sudo systemctl enable mssql-server
    sudo systemctl start mssql-server
    
    # Wait for SQL Server to start
    print_status "Waiting for SQL Server to start..."
    sleep 15
    
    print_success "SQL Server configured and started"
}

# Install SQL Server command line tools
install_sqlcmd() {
    print_status "Installing SQL Server command line tools..."
    
    # Install mssql-tools from AUR
    $AUR_HELPER -S --noconfirm mssql-tools
    
    # Add to PATH if not already there
    if ! echo $PATH | grep -q "/opt/mssql-tools/bin"; then
        echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
        export PATH="$PATH:/opt/mssql-tools/bin"
    fi
    
    print_success "SQL Server tools installed"
}

# Create database and user
setup_database() {
    print_status "Creating database and user..."
    
    # Create the database (with -C flag to trust server certificate)
    /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "
    IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = '$DATABASE_NAME')
    BEGIN
        CREATE DATABASE [$DATABASE_NAME]
    END
    "
    
    # Create the user (with -C flag to trust server certificate)
    /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "
    USE [$DATABASE_NAME]
    IF NOT EXISTS (SELECT name FROM sys.database_principals WHERE name = '$USER_NAME')
    BEGIN
        CREATE LOGIN [$USER_NAME] WITH PASSWORD = '$USER_PASSWORD'
        CREATE USER [$USER_NAME] FOR LOGIN [$USER_NAME]
        ALTER ROLE db_owner ADD MEMBER [$USER_NAME]
    END
    "
    
    print_success "Database '$DATABASE_NAME' and user '$USER_NAME' created"
}

# Create Python bulk import script based on your template
create_bulk_import_script() {
    print_status "Creating bulk data import script..."
    
    cat > ~/binance_bulk_import.py << EOF
#!/usr/bin/env python3
"""
Binance Vision Bulk Data Import for SQL Server
Based on the provided template for efficient bulk imports
"""

import os
import pyodbc
import time
import csv
import sys

# Database configuration
DRIVER = '{ODBC Driver 17 for SQL Server}'
SERVER = 'localhost'
DATABASE = '$DATABASE_NAME'
USERNAME = '$USER_NAME'
PASSWORD = '$USER_PASSWORD'

BATCH_SIZE = 250000

def table_exists(cursor, table_name):
    cursor.execute(f"""
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME = N'{table_name}'
    """)
    return cursor.fetchone()[0] > 0

def get_latest_timestamp(cursor, table_name):
    cursor.execute(f"SELECT MAX(TimestampEnd) FROM [{table_name}]")
    result = cursor.fetchone()[0]
    return result if result else 0

def bulk_import_binance_data(data_directory):
    """
    Bulk import Binance Vision data using the efficient template method
    """
    try:
        connection = pyodbc.connect(f'DRIVER={DRIVER};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};TrustServerCertificate=yes;')
        cursor = connection.cursor()
        print("Connected to the BinanceData database successfully")

        if not os.path.exists(data_directory):
            print(f"Error: Directory {data_directory} does not exist")
            return False

        folders = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]
        print(f"Found {len(folders)} folders")

        for folder in folders:
            pair_name = folder.replace('_', '')
            table_name = f'{pair_name}_klines'
            print(f"Checking table {table_name}")

            if not table_exists(cursor, table_name):
                print(f"Table {table_name} does not exist. Creating it.")
                create_table_sql = f"""
                CREATE TABLE [{table_name}] (
                    CandleID INT PRIMARY KEY IDENTITY(1,1),
                    Timeframe VARCHAR(10) NOT NULL,
                    TimestampStart BIGINT NOT NULL,
                    [Open] DECIMAL(28, 8) NOT NULL,
                    [High] DECIMAL(28, 8) NOT NULL,
                    [Low] DECIMAL(28, 8) NOT NULL,
                    [Close] DECIMAL(28, 8) NOT NULL,
                    Volume DECIMAL(28, 8) NOT NULL,
                    TimestampEnd BIGINT NOT NULL,
                    QuoteVolume DECIMAL(28, 8) NOT NULL,
                    Trades INT NOT NULL,
                    TakerBaseVolume DECIMAL(28, 8) NOT NULL,
                    TakerQuoteVolume DECIMAL(28, 8) NOT NULL
                )
                """
                cursor.execute(create_table_sql)
                print(f"Table {table_name} created successfully")
                latest_timestamp = 0
            else:
                print(f"Table {table_name} already exists. Appending data.")
                latest_timestamp = get_latest_timestamp(cursor, table_name)

            print(f"Processing table {table_name}")

            timeframe_folders = [f for f in os.listdir(os.path.join(data_directory, folder)) if os.path.isdir(os.path.join(data_directory, folder, f))]
            print(f"Processing {folder} with {len(timeframe_folders)} timeframes")

            for timeframe_folder in timeframe_folders:
                csv_files = sorted([f for f in os.listdir(os.path.join(data_directory, folder, timeframe_folder)) if f.endswith('.csv')])
                print(f"Found {len(csv_files)} CSV files for {timeframe_folder}")

                for csv_file in csv_files:
                    file_path = os.path.join(data_directory, folder, timeframe_folder, csv_file)
                    print(f"Processing {file_path}")

                    start_time = time.time()
                    rows_inserted = 0

                    with open(file_path, 'r') as file:
                        csv_reader = csv.reader(file)

                        sql = f"""INSERT INTO [{table_name}] (Timeframe,
                        TimestampStart, 
                        [Open], 
                        [High], 
                        [Low], 
                        [Close], 
                        Volume, 
                        TimestampEnd, 
                        QuoteVolume, 
                        Trades, 
                        TakerBaseVolume, 
                        TakerQuoteVolume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

                        batch = []
                        for row in csv_reader:
                            if int(row[0]) > latest_timestamp:
                                batch.append([timeframe_folder] + row[:11])
                                if len(batch) == BATCH_SIZE:
                                    cursor.executemany(sql, batch)
                                    rows_inserted += len(batch)
                                    elapsed_time = time.time() - start_time
                                    print(f"Inserted {rows_inserted} rows. Elapsed time: {elapsed_time:.2f} seconds")
                                    connection.commit()
                                    batch = []

                        if batch:
                            cursor.executemany(sql, batch)
                            rows_inserted += len(batch)
                            connection.commit()

                    print(f"Committed changes for {table_name}")
                    print(f"Total rows inserted: {rows_inserted}")
                    print(f"Total time for {csv_file}: {time.time() - start_time:.2f} seconds")

        return True

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False
    finally:
        if connection:
            connection.close()
            print("Connection closed")

def main():
    if len(sys.argv) < 2:
        print("Usage: python binance_bulk_import.py <data_directory>")
        print("Example: python binance_bulk_import.py /path/to/binance-data/spot/monthly/klines")
        return
    
    data_directory = sys.argv[1]
    
    print("Starting bulk import of Binance Vision data...")
    success = bulk_import_binance_data(data_directory)
    
    if success:
        print("Script execution completed successfully!")
    else:
        print("Script execution failed!")

if __name__ == "__main__":
    main()
EOF

    chmod +x ~/binance_bulk_import.py
    print_success "Bulk import script created at ~/binance_bulk_import.py"
}

# Install custom C++ driver
install_custom_driver() {
    print_status "Installing custom C++ driver from BTQuant..."
    
    # Check if BTQuant directory exists
    if [ ! -d "dependencies" ]; then
        print_status "BTQuant directory not found. Cloning repository..."
        git clone https://github.com/itsXactlY/BTQuant/ BTQuant || {
            print_error "Failed to clone BTQuant repository"
            return 1
        }
    fi
    
    # Check if setup.py exists
    if [ ! -f "dependencies/MsSQL/setup.py" ]; then
        print_error "Custom driver setup.py not found at BTQuant/dependencies/MsSQL/setup.py"
        return 1
    fi
    
    # Install the custom driver
    cd dependencies/MsSQL
    python3 setup.py install || {
        print_error "Failed to install custom C++ driver"
        cd - > /dev/null
        return 1
    }
    cd - > /dev/null
    
    print_success "Custom C++ driver installed successfully"
}

# Create credentials file
create_credentials_file() {
    print_status "Creating credentials file..."
    
    cat > ~/mssql_credentials.txt << EOF
=== Microsoft SQL Server Credentials ===
Generated on: $(date)

Server: localhost
Database: $DATABASE_NAME

=== Administrator Account ===
Username: sa
Password: $MSSQL_SA_PASSWORD

=== Binance User Account ===
Username: $USER_NAME
Password: $USER_PASSWORD

=== Connection Strings ===
SA Connection: 
sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C

Binance User Connection:
sqlcmd -S localhost -U $USER_NAME -P "$USER_PASSWORD" -d $DATABASE_NAME -C

Python Connection String:
DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=$DATABASE_NAME;UID=$USER_NAME;PWD=$USER_PASSWORD;TrustServerCertificate=yes

=== Service Commands ===
Start:   sudo systemctl start mssql-server
Stop:    sudo systemctl stop mssql-server
Restart: sudo systemctl restart mssql-server
Status:  sudo systemctl status mssql-server

=== Import Scripts ===
Bulk Import: python ~/binance_bulk_import.py /path/to/data

=== Important Notes ===
- Keep this file secure - it contains sensitive passwords
- SQL Server edition: Developer (free, non-production use)
- Language: English
- Database is ready for Binance Vision data import

EOF

    # Set secure permissions on credentials file
    chmod 600 ~/mssql_credentials.txt
    
    print_success "Credentials file created at ~/mssql_credentials.txt"
    print_warning "Credentials file permissions set to 600 (owner read/write only)"
}

# Run data import if requested
run_data_import() {
    if [ -n "$DATA_IMPORT_PATH" ] && [ -d "$DATA_IMPORT_PATH" ]; then
        print_status "Starting bulk data import..."
        echo "This may take a while depending on data size..."
        
        python3 ~/binance_bulk_import.py "$DATA_IMPORT_PATH"
        
        if [ $? -eq 0 ]; then
            print_success "Data import completed successfully!"
        else
            print_warning "Data import encountered errors. Check the output above."
        fi
    fi
}

# Display connection info
show_connection_info() {
    print_success "=== SQL Server Installation Complete ==="
    echo
    print_status "Installation Summary:"
    echo "  ✓ SQL Server Developer Edition installed"
    echo "  ✓ Database '$DATABASE_NAME' created"
    echo "  ✓ User '$USER_NAME' created with full permissions"
    echo "  ✓ Bulk import script ready"
    echo "  ✓ Credentials saved to ~/mssql_credentials.txt"
    echo
    print_status "Quick Start:"
    echo "  View credentials: cat ~/mssql_credentials.txt"
    echo "  Connect with SA:  sqlcmd -S localhost -U sa -P '[password_from_file]'"
    echo "  Connect with user: sqlcmd -S localhost -U $USER_NAME -P '[password_from_file]' -d $DATABASE_NAME"
    echo "  Import data: python ~/binance_bulk_import.py /path/to/data"
    echo
    print_warning "IMPORTANT: Your credentials are saved in ~/mssql_credentials.txt"
    print_warning "Keep this file secure - it contains sensitive passwords!"
    echo
    print_warning "Remember to source your bashrc or restart terminal for PATH changes:"
    echo "  source ~/.bashrc"
}

# Complete uninstall and purge of SQL Server
purge_sqlserver() {
    print_status "Completely removing existing SQL Server installation..."
    
    # Stop SQL Server service if running
    print_status "Stopping SQL Server service..."
    sudo systemctl stop mssql-server 2>/dev/null || true
    sudo systemctl disable mssql-server 2>/dev/null || true
    
    # Remove AUR packages
    print_status "Removing SQL Server packages..."
    if command_exists yay; then
        yay -Rns --noconfirm mssql-server mssql-tools 2>/dev/null || true
    elif command_exists paru; then
        paru -Rns --noconfirm mssql-server mssql-tools 2>/dev/null || true
    else
        sudo pacman -Rns --noconfirm mssql-server mssql-tools 2>/dev/null || true
    fi
    
    # Remove SQL Server directories and files
    print_status "Removing SQL Server data and configuration..."
    sudo rm -rf /opt/mssql/ 2>/dev/null || true
    sudo rm -rf /var/opt/mssql/ 2>/dev/null || true
    sudo rm -rf /etc/systemd/system/mssql-server.service 2>/dev/null || true
    sudo rm -rf /usr/lib/systemd/system/mssql-server.service 2>/dev/null || true
    
    # Remove any leftover configuration
    sudo rm -rf ~/.mssql/ 2>/dev/null || true
    sudo rm -rf /tmp/mssql* 2>/dev/null || true
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Remove old scripts and credentials
    rm -f ~/binance_bulk_import.py 2>/dev/null || true
    rm -f ~/mssql_credentials.txt 2>/dev/null || true
    
    print_success "SQL Server completely removed"
}

# Main installation function
main() {
    print_status "Starting Microsoft SQL Server installation for Arch Linux..."
    echo
    
    # Ask if user wants to purge existing installation
    echo -n "Do you want to completely remove any existing SQL Server installation first? (recommended if reinstalling) (y/n): "
    read -r purge_response
    if [[ $purge_response =~ ^[Yy]$ ]]; then
        purge_sqlserver
        echo
    fi
    
    detect_distro
    install_aur_helper
    install_dependencies
    get_user_input
    install_sqlserver
    configure_sqlserver
    install_sqlcmd
    setup_database
    create_credentials_file
    create_bulk_import_script
    # install_custom_driver
    run_data_import
    
    show_connection_info
    
    print_success "Installation completed successfully!"
}

# Run main function
main "$@"