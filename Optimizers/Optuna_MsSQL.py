# optuna_db_setup.py
from backtrader.dontcommit import fast_mssql

def create_optuna_tables(connection_string):
    """Create Optuna schema tables in MSSQL database"""
    
    # SQL to create Optuna tables (based on Optuna's schema)
    create_tables_sql = [
        """
        IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'OptunaBT')
        BEGIN
            CREATE DATABASE OptunaBT;
        END
        """,
        """
        USE OptunaBT;
        
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'studies')
        BEGIN
            CREATE TABLE studies (
                study_id INT IDENTITY(1,1) PRIMARY KEY,
                study_name NVARCHAR(512) NOT NULL UNIQUE,
                direction INT NOT NULL,
                user_attrs TEXT,
                system_attrs TEXT
            );
        END
        """,
        """
        USE OptunaBT;
        
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'trials')
        BEGIN
            CREATE TABLE trials (
                trial_id INT IDENTITY(1,1) PRIMARY KEY,
                number INT NOT NULL,
                study_id INT NOT NULL,
                state INT NOT NULL,
                value FLOAT,
                datetime_start DATETIME2,
                datetime_complete DATETIME2,
                params TEXT,
                distributions TEXT,
                user_attrs TEXT,
                system_attrs TEXT,
                intermediate_values TEXT,
                FOREIGN KEY (study_id) REFERENCES studies(study_id) ON DELETE CASCADE,
                UNIQUE (study_id, number)
            );
        END
        """,
        """
        USE OptunaBT;
        
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'trial_params')
        BEGIN
            CREATE TABLE trial_params (
                param_id INT IDENTITY(1,1) PRIMARY KEY,
                trial_id INT NOT NULL,
                param_name NVARCHAR(512) NOT NULL,
                param_value FLOAT NOT NULL,
                distribution TEXT NOT NULL,
                FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
            );
        END
        """,
        """
        USE OptunaBT;
        
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'trial_values')
        BEGIN
            CREATE TABLE trial_values (
                trial_value_id INT IDENTITY(1,1) PRIMARY KEY,
                trial_id INT NOT NULL,
                objective INT NOT NULL,
                value FLOAT,
                FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
            );
        END
        """,
        """
        USE OptunaBT;
        
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'trial_user_attributes')
        BEGIN
            CREATE TABLE trial_user_attributes (
                trial_user_attribute_id INT IDENTITY(1,1) PRIMARY KEY,
                trial_id INT NOT NULL,
                key_name NVARCHAR(512) NOT NULL,
                value_json TEXT NOT NULL,
                FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
                UNIQUE (trial_id, key_name)
            );
        END
        """,
        """
        USE OptunaBT;
        
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'trial_system_attributes')
        BEGIN
            CREATE TABLE trial_system_attributes (
                trial_system_attribute_id INT IDENTITY(1,1) PRIMARY KEY,
                trial_id INT NOT NULL,
                key_name NVARCHAR(512) NOT NULL,
                value_json TEXT NOT NULL,
                FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE,
                UNIQUE (trial_id, key_name)
            );
        END
        """
    ]
    
    for sql in create_tables_sql:
        try:
            fast_mssql.execute_non_query(connection_string, sql)
            print(f"✓ Executed SQL command successfully")
        except Exception as e:
            print(f"✗ Error executing SQL: {e}")
            print(f"SQL: {sql[:100]}...")

if __name__ == "__main__":
    # Your MSSQL connection string
    CONNECTION_STRING = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=BinanceData;UID=SA;PWD=q?}33YIToo:H%xue$Kr*;TrustServerCertificate=yes;"
    
    create_optuna_tables(CONNECTION_STRING)
    print("Optuna database schema created successfully!")