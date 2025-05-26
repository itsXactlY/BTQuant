#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sql.h>
#include <sqlext.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace py = pybind11;

class ODBCManager {
public:
    ODBCManager(const std::string& connection_string) {
        SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env);
        SQLSetEnvAttr(env, SQL_ATTR_ODBC_VERSION, (void*)SQL_OV_ODBC3, 0);
        
        SQLAllocHandle(SQL_HANDLE_DBC, env, &dbc);
        
        // Connect to data source
        SQLRETURN ret = SQLDriverConnect(dbc, NULL, (SQLCHAR*)connection_string.c_str(), SQL_NTS,
                                        NULL, 0, NULL, SQL_DRIVER_NOPROMPT);
        if (!SQL_SUCCEEDED(ret)) {
            cleanup();
            throw std::runtime_error("Failed to connect to database");
        }
        
        SQLAllocHandle(SQL_HANDLE_STMT, dbc, &stmt);
    }

    ~ODBCManager() {
        cleanup();
    }

    void resetStatement() {
        if (stmt != SQL_NULL_HANDLE) {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
        }
        SQLAllocHandle(SQL_HANDLE_STMT, dbc, &stmt);
    }

    void executeQuery(const std::string& query) {
        SQLRETURN ret = SQLExecDirect(stmt, (SQLCHAR*)query.c_str(), SQL_NTS);
        if (!SQL_SUCCEEDED(ret)) {
            SQLCHAR sqlstate[6], message[SQL_MAX_MESSAGE_LENGTH];
            SQLINTEGER native_error;
            SQLSMALLINT length;
            SQLGetDiagRec(SQL_HANDLE_STMT, stmt, 1, sqlstate, &native_error, message, sizeof(message), &length);
            std::string error_message = "Failed to execute query: " + std::string((char*)message);
            throw std::runtime_error(error_message);
        }
    }

    std::vector<std::vector<std::string>> fetchData() {
        // Get column count
        SQLSMALLINT columnCount;
        SQLNumResultCols(stmt, &columnCount);
        
        std::vector<std::vector<std::string>> data;
        SQLLEN indicator;
        char buffer[1024];
        
        // Fetch and store results
        while (SQL_SUCCEEDED(SQLFetch(stmt))) {
            std::vector<std::string> row;
            for (SQLSMALLINT i = 1; i <= columnCount; i++) {
                SQLRETURN ret = SQLGetData(stmt, i, SQL_C_CHAR, buffer, sizeof(buffer), &indicator);
                if (SQL_SUCCEEDED(ret) && indicator != SQL_NULL_DATA) {
                    row.push_back(std::string(buffer));
                } else {
                    row.push_back("NULL");
                }
            }
            data.push_back(row);
        }
        return data;
    }

    bool isConnected() {
        if (dbc == SQL_NULL_HANDLE) return false;
        
        SQLINTEGER dead;
        SQLRETURN ret = SQLGetConnectAttr(dbc, SQL_ATTR_CONNECTION_DEAD, &dead, 0, NULL);
        return SQL_SUCCEEDED(ret) && dead == SQL_CD_FALSE;
    }

private:
    SQLHENV env = SQL_NULL_HANDLE;
    SQLHDBC dbc = SQL_NULL_HANDLE;
    SQLHSTMT stmt = SQL_NULL_HANDLE;

    void cleanup() {
        if (stmt != SQL_NULL_HANDLE) {
            SQLFreeHandle(SQL_HANDLE_STMT, stmt);
            stmt = SQL_NULL_HANDLE;
        }
        if (dbc != SQL_NULL_HANDLE) {
            SQLDisconnect(dbc);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            dbc = SQL_NULL_HANDLE;
        }
        if (env != SQL_NULL_HANDLE) {
            SQLFreeHandle(SQL_HANDLE_ENV, env);
            env = SQL_NULL_HANDLE;
        }
    }
};

class ConnectionPool {
private:
    static std::unordered_map<std::string, std::shared_ptr<ODBCManager>> connections;
    static std::mutex pool_mutex;
    
public:
    static std::shared_ptr<ODBCManager> getConnection(const std::string& connection_string) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        
        auto it = connections.find(connection_string);
        if (it != connections.end() && it->second && it->second->isConnected()) {
            return it->second;
        }
        
        try {
            connections[connection_string] = std::make_shared<ODBCManager>(connection_string);
            return connections[connection_string];
        } catch (const std::exception& e) {
            connections.erase(connection_string);
            throw;
        }
    }
    
    static void closeAll() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        connections.clear();
    }
    
    static size_t getPoolSize() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        return connections.size();
    }
    
    static void removeConnection(const std::string& connection_string) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        connections.erase(connection_string);
    }
};

std::unordered_map<std::string, std::shared_ptr<ODBCManager>> ConnectionPool::connections;
std::mutex ConnectionPool::pool_mutex;

std::vector<std::vector<std::string>> fetch_data_from_db(const std::string& connection_string,
                                                        const std::string& query) {
    try {
        auto odbc = ConnectionPool::getConnection(connection_string);
        odbc->resetStatement();
        odbc->executeQuery(query);
        return odbc->fetchData();
    } catch (const std::runtime_error& e) {
        ConnectionPool::removeConnection(connection_string);
        throw std::runtime_error(e.what());
    }
}

PYBIND11_MODULE(fast_mssql, m) {
    m.doc() = "Fast MSSQL driver with connection pooling";
    
    m.def("fetch_data_from_db", &fetch_data_from_db, 
          "Fetch data from MSSQL database with connection pooling",
          py::arg("connection_string"), py::arg("query"));
          
    m.def("close_all_connections", &ConnectionPool::closeAll, 
          "Close all pooled connections");
          
    m.def("get_pool_size", &ConnectionPool::getPoolSize, 
          "Get current connection pool size");
          
    m.def("remove_connection", &ConnectionPool::removeConnection, 
          "Remove specific connection from pool",
          py::arg("connection_string"));
}