# debug_mssql_connection.py
from backtrader.dontcommit import fast_mssql, connection_string

def test_connection_strings():
    """Test different connection string formats"""
    
    # Test connection strings - try these one by one
    connection_strings = [
        # Format 1: Current format
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=master;UID=SA;PWD=YourSuperSecret!Password;TrustServerCertificate=yes",
        
        # Format 2: Different driver name
        "DRIVER={SQL Server};SERVER=localhost;DATABASE=master;UID=SA;PWD=YourSuperSecret!Password;TrustServerCertificate=yes",
        
        # Format 3: Using instance name
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=master;UID=SA;PWD=YourSuperSecret!Password;TrustServerCertificate=yes",
        
        # Format 4: Using port
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost,1433;DATABASE=master;UID=SA;PWD=YourSuperSecret!Password;TrustServerCertificate=yes",
        
        # Format 5: Trusted connection (Windows auth)
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=master;Trusted_Connection=yes;TrustServerCertificate=yes",

        connection_string
    ]
    
    for i, conn_str in enumerate(connection_strings, 1):
        print(f"\n--- Testing Connection String {i} ---")
        print(f"Connection: {conn_str[:50]}...")
        
        try:
            # Test simple query
            result = fast_mssql.fetch_data_from_db(conn_str, "SELECT @@VERSION")
            print(f"✓ SUCCESS! Connected to SQL Server")
            print(f"Version: {result[0][0][:100]}...")
            return conn_str
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
    
    return None

def test_create_database(connection_string):
    """Test creating database with working connection"""
    try:
        # First, try to create the database
        create_db_sql = """
        IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'OptunaBT')
        BEGIN
            CREATE DATABASE OptunaBT;
        END
        """
        
        print("\n--- Testing Database Creation ---")
        fast_mssql.execute_non_query(connection_string, create_db_sql)
        print("✓ Database creation command executed")
        
        # Test if database exists
        check_db_sql = "SELECT name FROM sys.databases WHERE name = 'OptunaBT'"
        result = fast_mssql.fetch_data_from_db(connection_string, check_db_sql)
        
        if result:
            print("✓ OptunaBT database exists")
            return True
        else:
            print("✗ OptunaBT database not found")
            return False
            
    except Exception as e:
        print(f"✗ Database creation failed: {e}")
        return False

if __name__ == "__main__":
    print("=== MSSQL Connection Debugger ===")
    
    # Step 1: Test connection
    working_conn = test_connection_strings()
    
    if working_conn:
        print(f"\n🎉 Found working connection string!")
        print(f"Use this: {working_conn}")
        
        # Step 2: Test database creation
        if test_create_database(working_conn):
            print("\n✓ Ready to create Optuna tables!")
        else:
            print("\n✗ Database creation issues - check permissions")
    else:
        print("\n❌ No working connection found!")
        print("\nTroubleshooting steps:")
        print("1. Check if SQL Server is running: `sudo systemctl status mssql-server`")
        print("2. Check if SA account is enabled")
        print("3. Verify password and server name")
        print("4. Check firewall settings")
        print("5. Try connecting with sqlcmd: `sqlcmd -S localhost -U SA -P 'your_password'`")

