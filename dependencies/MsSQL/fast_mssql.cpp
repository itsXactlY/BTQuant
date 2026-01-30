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
<<<<<<< HEAD
#include <chrono>
#include <iostream>

namespace py = pybind11;

// ==========================
// ODBC Manager Class
// ==========================

class ODBCManager {
public:
    ODBCManager(const std::string& connection_string) 
        : connection_string_(connection_string) {
        
        // Allocate environment handle
        SQLRETURN ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env);
        if (!SQL_SUCCEEDED(ret)) {
            throw std::runtime_error("SQLAllocHandle ENV failed");
        }

        // Set ODBC version
        ret = SQLSetEnvAttr(env, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, 0);
        if (!SQL_SUCCEEDED(ret)) {
            cleanup();
            throw std::runtime_error("SQLSetEnvAttr ODBC_VERSION failed");
        }

        // Allocate connection handle
        ret = SQLAllocHandle(SQL_HANDLE_DBC, env, &dbc);
        if (!SQL_SUCCEEDED(ret)) {
            cleanup();
            throw std::runtime_error("SQLAllocHandle DBC failed");
        }

        // Connect to database
        ret = SQLDriverConnect(dbc, NULL, (SQLCHAR*)connection_string.c_str(), SQL_NTS,
                              NULL, 0, NULL, SQL_DRIVER_NOPROMPT);
        if (!SQL_SUCCEEDED(ret)) {
            cleanup();
            throw std::runtime_error("SQLDriverConnect failed: " + getLastError(SQL_HANDLE_DBC, dbc));
        }

        //  CRITICAL: Start with autocommit OFF for transaction control
        SQLSetConnectAttr(dbc, SQL_ATTR_AUTOCOMMIT, (SQLPOINTER)SQL_AUTOCOMMIT_OFF, SQL_IS_UINTEGER);
        
        // Allocate statement handle
=======

namespace py = pybind11;

class ODBCManager {
public:
    ODBCManager(const std::string& connection_string) {
        SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env);
        SQLSetEnvAttr(env, SQL_ATTR_ODBC_VERSION, (void*)SQL_OV_ODBC3, 0);

        SQLAllocHandle(SQL_HANDLE_DBC, env, &dbc);

        // Connect
        SQLRETURN ret = SQLDriverConnect(dbc, NULL, (SQLCHAR*)connection_string.c_str(), SQL_NTS,
                                        NULL, 0, NULL, SQL_DRIVER_NOPROMPT);
        if (!SQL_SUCCEEDED(ret)) {
            cleanup();
            throw std::runtime_error("Failed to connect to database");
        }

>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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

<<<<<<< HEAD
    //  NEW: Transaction control methods
    void beginTransaction() {
        // Verify connection is healthy before starting transaction
        if (!isHealthy()) {
            throw std::runtime_error("Connection not healthy for transaction");
        }
        // Already in transaction mode due to AUTOCOMMIT_OFF
    }

    void commitTransaction() {
        SQLRETURN ret = SQLEndTran(SQL_HANDLE_DBC, dbc, SQL_COMMIT);
        if (!SQL_SUCCEEDED(ret)) {
            throwSQLDBError("Commit failed");
        }
        // Stay in transaction mode
        SQLSetConnectAttr(dbc, SQL_ATTR_AUTOCOMMIT, (SQLPOINTER)SQL_AUTOCOMMIT_OFF, SQL_IS_UINTEGER);
    }

    void rollbackTransaction() {
        SQLRETURN ret = SQLEndTran(SQL_HANDLE_DBC, dbc, SQL_ROLLBACK);
        if (!SQL_SUCCEEDED(ret)) {
            throwSQLDBError("Rollback failed");
        }
        SQLSetConnectAttr(dbc, SQL_ATTR_AUTOCOMMIT, (SQLPOINTER)SQL_AUTOCOMMIT_OFF, SQL_IS_UINTEGER);
    }

=======
    // Run SELECT / query with result set
    void executeQuery(const std::string& query) {
        SQLRETURN ret = SQLExecDirect(stmt, (SQLCHAR*)query.c_str(), SQL_NTS);
        if (!SQL_SUCCEEDED(ret)) {
            throwSQLStmtError("Failed to execute query");
        }
    }

    // Run non-query (CREATE, DROP, UPDATE, DELETE…)
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
    void executeNonQuery(const std::string& query) {
        resetStatement();
        SQLRETURN ret = SQLExecDirect(stmt, (SQLCHAR*)query.c_str(), SQL_NTS);
        if (!SQL_SUCCEEDED(ret)) {
            throwSQLStmtError("Failed to execute non-query");
        }
    }

<<<<<<< HEAD
    //  FIXED: True bulk insert with transaction control
    void bulkInsert(const std::string& query, 
                    const std::vector<std::vector<std::string>>& rows) {
        if (rows.empty()) return;
        
=======
    // Executemany-style bulk insert
    void bulkInsert(const std::string& query, const std::vector<std::vector<std::string>>& rows) {
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
        resetStatement();

        SQLRETURN ret = SQLPrepare(stmt, (SQLCHAR*)query.c_str(), SQL_NTS);
        if (!SQL_SUCCEEDED(ret)) {
            throwSQLStmtError("Failed to prepare bulk insert");
        }

<<<<<<< HEAD
        //  Begin transaction ONCE
        beginTransaction();

=======
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
        for (const auto& row : rows) {
            std::vector<SQLLEN> indicators(row.size(), SQL_NTS);
            std::vector<const char*> cstrs;
            cstrs.reserve(row.size());
            for (auto& val : row) cstrs.push_back(val.c_str());

            for (size_t i = 0; i < row.size(); i++) {
<<<<<<< HEAD
                //  Handle VARCHAR(MAX) correctly with SQL_LONGVARCHAR
                SQLSMALLINT sql_type = (row[i].size() > 8000) ? SQL_LONGVARCHAR : SQL_VARCHAR;
                SQLULEN precision = (row[i].size() > 8000) ? 0 : row[i].size();

                SQLBindParameter(stmt, (SQLUSMALLINT)(i + 1), SQL_PARAM_INPUT, SQL_C_CHAR,
                                 sql_type, precision, 0,
=======
                SQLBindParameter(stmt, (SQLUSMALLINT)(i + 1), SQL_PARAM_INPUT, SQL_C_CHAR,
                                 SQL_VARCHAR, row[i].size(), 0,
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
                                 (SQLPOINTER)cstrs[i], row[i].size(), &indicators[i]);
            }

            ret = SQLExecute(stmt);
<<<<<<< HEAD
            //  Only throw on real errors, not SUCCESS_WITH_INFO
            if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
                rollbackTransaction();
                throwSQLStmtError("Bulk insert failed during row execution");
            }
            
            SQLFreeStmt(stmt, SQL_RESET_PARAMS);
        }

        //  Commit ONCE at end
        commitTransaction();
    }

    //  NEW: Health check
    bool isHealthy() {
        if (dbc == SQL_NULL_HANDLE) return false;
        
        SQLHSTMT test_stmt;
        SQLAllocHandle(SQL_HANDLE_STMT, dbc, &test_stmt);
        
        SQLRETURN ret = SQLExecDirect(test_stmt, (SQLCHAR*)"SELECT 1", SQL_NTS);
        
        SQLFreeHandle(SQL_HANDLE_STMT, test_stmt);
        
        return SQL_SUCCEEDED(ret);
=======
            if (!SQL_SUCCEEDED(ret)) {
                throwSQLStmtError("Bulk insert failed during row execution");
            }
        }
    }

    std::vector<std::vector<std::string>> fetchData() {
        // Get column count
        SQLSMALLINT columnCount;
        SQLNumResultCols(stmt, &columnCount);

        std::vector<std::vector<std::string>> data;
        SQLLEN indicator;
        char buffer[1024];

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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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
<<<<<<< HEAD
    std::string connection_string_;
=======
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

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

<<<<<<< HEAD
    //  NEW: DB error handler
    void throwSQLDBError(const std::string& prefix) {
        SQLCHAR sqlstate[6], message[SQL_MAX_MESSAGE_LENGTH];
        SQLINTEGER native_error;
        SQLSMALLINT length;
        SQLGetDiagRec(SQL_HANDLE_DBC, dbc, 1, sqlstate, &native_error, message, sizeof(message), &length);
        throw std::runtime_error(prefix + ": [" + std::string((char*)sqlstate) + "] " + std::string((char*)message));
    }

    std::string getLastError(SQLSMALLINT handle_type, SQLHANDLE handle) {
        SQLCHAR sqlstate[6], message[SQL_MAX_MESSAGE_LENGTH];
        SQLINTEGER native_error;
        SQLSMALLINT length;
        SQLGetDiagRec(handle_type, handle, 1, sqlstate, &native_error, message, sizeof(message), &length);
        return std::string((char*)sqlstate) + "] " + std::string((char*)message);
    }

    void throwSQLStmtError(const std::string& prefix) {
        throw std::runtime_error(prefix + ": " + getLastError(SQL_HANDLE_STMT, stmt));
    }
};

// ==========================
// Connection Pool Class
// ==========================

=======
    void throwSQLStmtError(const std::string& prefix) {
        SQLCHAR sqlstate[6], message[SQL_MAX_MESSAGE_LENGTH];
        SQLINTEGER native_error;
        SQLSMALLINT length;
        SQLGetDiagRec(SQL_HANDLE_STMT, stmt, 1, sqlstate, &native_error, message, sizeof(message), &length);
        throw std::runtime_error(prefix + ": " + std::string((char*)message));
    }
};

>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
class ConnectionPool {
private:
    static std::unordered_map<std::string, std::shared_ptr<ODBCManager>> connections;
    static std::mutex pool_mutex;

public:
    static std::shared_ptr<ODBCManager> getConnection(const std::string& connection_string) {
        std::lock_guard<std::mutex> lock(pool_mutex);

        auto it = connections.find(connection_string);
<<<<<<< HEAD
        if (it != connections.end() && it->second && it->second->isHealthy()) {
=======
        if (it != connections.end() && it->second && it->second->isConnected()) {
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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

// ==========================
// Python bindings
// ==========================

std::vector<std::vector<std::string>> fetch_data_from_db(const std::string& connection_string,
                                                        const std::string& query) {
    auto odbc = ConnectionPool::getConnection(connection_string);
    odbc->resetStatement();
    odbc->executeQuery(query);
    return odbc->fetchData();
}

PYBIND11_MODULE(fast_mssql, m) {
<<<<<<< HEAD
    m.doc() = "Fast MSSQL driver with transaction control and connection pooling";

    py::class_<ODBCManager>(m, "ODBCManager")
        .def(py::init<const std::string&>())
        .def("execute_non_query", &ODBCManager::executeNonQuery)
        .def("bulk_insert", &ODBCManager::bulkInsert)
        .def("begin_transaction", &ODBCManager::beginTransaction)
        .def("commit_transaction", &ODBCManager::commitTransaction)
        .def("rollback_transaction", &ODBCManager::rollbackTransaction)
        .def("is_healthy", &ODBCManager::isHealthy)
        .def("is_connected", &ODBCManager::isConnected);
=======
    m.doc() = "Fast MSSQL driver with connection pooling and bulk insert (v2)";
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

    m.def("fetch_data_from_db", &fetch_data_from_db,
          "Fetch data from MSSQL database with connection pooling",
          py::arg("connection_string"), py::arg("query"));

    m.def("execute_non_query", [](const std::string& conn_str, const std::string& query) {
        auto odbc = ConnectionPool::getConnection(conn_str);
        odbc->executeNonQuery(query);
    }, "Execute non-query SQL (CREATE, DROP, UPDATE, DELETE)");

    m.def("bulk_insert", [](const std::string& conn_str, const std::string& query,
                            const std::vector<std::vector<std::string>>& rows) {
        auto odbc = ConnectionPool::getConnection(conn_str);
        odbc->bulkInsert(query, rows);
<<<<<<< HEAD
    }, "Perform bulk insert with transaction control");
=======
    }, "Perform bulk insert into a table");
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

    m.def("close_all_connections", &ConnectionPool::closeAll,
          "Close all pooled connections");

    m.def("get_pool_size", &ConnectionPool::getPoolSize,
          "Get current connection pool size");

    m.def("remove_connection", &ConnectionPool::removeConnection,
          "Remove specific connection from pool",
          py::arg("connection_string"));
}
