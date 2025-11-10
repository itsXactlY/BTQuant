import fast_mssql

# Test connection pooling
pool_size = fast_mssql.get_pool_size()
print(f"Pool size: {pool_size}")  # Should be 0 initially

# Test basic query (requires SQL Server running)
conn_str = (
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=localhost;'
    'DATABASE=master;'
    'UID=SA;'
    'PWD=your_password;'
    'TrustServerCertificate=yes;'
)

result = fast_mssql.fetch_data_from_db(conn_str, "SELECT @@VERSION")
print(result)  # SQL Server version