#ifdef _WIN32
    #include <windows.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sql.h>
#include <sqlext.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace py = pybind11;

class ODBCManager {
public:
    ODBCManager(const std::string& connection_string) {
        // Allocate environment handle
        SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env);
        SQLSetEnvAttr(env, SQL_ATTR_ODBC_VERSION, (void*)SQL_OV_ODBC3, 0);

        // Allocate connection handle
        SQLAllocHandle(SQL_HANDLE_DBC, env, &dbc);

        // Connect to data source
        SQLRETURN ret = SQLDriverConnect(dbc, NULL, (SQLCHAR*)connection_string.c_str(), SQL_NTS,
                                         NULL, 0, NULL, SQL_DRIVER_NOPROMPT);
        if (!SQL_SUCCEEDED(ret)) {
            cleanup();
            throw std::runtime_error("Failed to connect to database");
        }

        // Allocate statement handle
        SQLAllocHandle(SQL_HANDLE_STMT, dbc, &stmt);
    }

    ~ODBCManager() {
        cleanup();
    }

    void executeQuery(const std::string& query) {
        SQLRETURN ret = SQLExecDirect(stmt, (SQLCHAR*)query.c_str(), SQL_NTS);
        if (!SQL_SUCCEEDED(ret)) {
            SQLCHAR sqlstate[6], message[SQL_MAX_MESSAGE_LENGTH];
            SQLINTEGER native_error;
            SQLSMALLINT length;
            SQLGetDiagRec(SQL_HANDLE_STMT, stmt, 1, sqlstate, &native_error, message, sizeof(message), &length);
            std::string error_message = "Failed to execute query: " + std::string((char*)message);
            cleanup();
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

std::vector<std::vector<std::string>> fetch_data_from_db(const std::string& connection_string, 
                                                         const std::string& query) {
    try {
        ODBCManager odbc(connection_string);
        odbc.executeQuery(query);
        return odbc.fetchData();
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(e.what());
    }
}

PYBIND11_MODULE(fast_mssql, m) {
    m.def("fetch_data_from_db", &fetch_data_from_db, "A function that fetches data from MSSQL");
}