#include "mssql_bulk_inserter.h"

#include <array>
#include <cstring>
#include <iostream>
#include <mutex>
#include <cctype>
#include <sql.h>
#include <sqlext.h>
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace MarketData;

namespace {

// Helper function for timestamped logging
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::tm tm = *std::localtime(&now_time);
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
    
    char ms_buffer[10];
    snprintf(ms_buffer, sizeof(ms_buffer), "%03d", static_cast<int>(now_ms.count()));
    
    return std::string(buffer) + "." + ms_buffer;
}

constexpr SQLULEN     kTimestampColumnSize   = 27;
constexpr SQLSMALLINT kTimestampScaleMicros  = 6;
constexpr std::size_t kTimestampStringLength = 32;

template <std::size_t N>
inline void copyStrFixed(const std::string& s, std::array<SQLCHAR, N>& buf) {
    static_assert(N >= 1, "buffer must have at least 1 byte");
    const std::size_t max_len = N - 1;
    const std::size_t len     = s.size() > max_len ? max_len : s.size();
    if (len) std::memcpy(buf.data(), s.data(), len);
    buf[len] = 0;
}

} // anonymous namespace

MSSQLBulkInserter::MSSQLBulkInserter(const std::string& connection_string, bool debug_mode)
    : connection_string_(connection_string), debug_mode_(debug_mode) {
    SQLRETURN ret;

    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Initializing database connection" << std::endl;
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Connection string: " << connection_string_ << std::endl;
    }

    // ENV
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Allocating ODBC environment handle" << std::endl;
    }
    ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env_);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to allocate ODBC environment handle" << std::endl;
        throw std::runtime_error("SQLAllocHandle ENV");
    }
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   ODBC environment handle allocated successfully" << std::endl;
    }

    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Setting ODBC version to ODBC3" << std::endl;
    }
    ret = SQLSetEnvAttr(env_, SQL_ATTR_ODBC_VERSION,
                        (SQLPOINTER)SQL_OV_ODBC3, 0);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to set ODBC version" << std::endl;
        SQLFreeHandle(SQL_HANDLE_ENV, env_);
        throw std::runtime_error("SQLSetEnvAttr ODBC_VERSION");
    }
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   ODBC version set successfully" << std::endl;
    }

    // DBC
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Allocating ODBC connection handle" << std::endl;
    }
    ret = SQLAllocHandle(SQL_HANDLE_DBC, env_, &dbc_);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to allocate ODBC connection handle" << std::endl;
        SQLFreeHandle(SQL_HANDLE_ENV, env_);
        throw std::runtime_error("SQLAllocHandle DBC");
    }
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   ODBC connection handle allocated successfully" << std::endl;
    }

    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Connecting to database using SQLDriverConnect" << std::endl;
    }
    SQLCHAR out_conn_str[1024];
    SQLSMALLINT out_len = 0;
    ret = SQLDriverConnect(
        dbc_, nullptr,
        (SQLCHAR*)connection_string_.c_str(),
        SQL_NTS,
        out_conn_str, sizeof(out_conn_str), &out_len,
        SQL_DRIVER_NOPROMPT);

    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Database connection failed" << std::endl;
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLDriverConnect");
    }

    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Database connection established successfully" << std::endl;
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Connection output: " << std::string((char*)out_conn_str, out_len) << std::endl;
    }

    // STMT used for data-path & DDL
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Allocating ODBC statement handle" << std::endl;
    }
    ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc_, &stmt_);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to allocate ODBC statement handle" << std::endl;
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLAllocHandle STMT");
    }
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   ODBC statement handle allocated successfully" << std::endl;
    }

    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Configuring database connection settings" << std::endl;
    }
    setAutoCommit(false);
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Auto-commit disabled, transaction mode enabled" << std::endl;
    }

    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Ensuring core database tables exist" << std::endl;
    }
    ensureCoreTables();
    
    // Add database connection verification if requested
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Verifying database connection..." << std::endl;
        if (verifyConnection()) {
            std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Database connection verified successfully" << std::endl;
        } else {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Database connection verification failed!" << std::endl;
            throw std::runtime_error("Database connection verification failed");
        }
    }
    
    // Test table creation if requested
    if (debug_mode_) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Testing table creation..." << std::endl;
        testTableCreation();
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Database initialization completed successfully" << std::endl;
}

MSSQLBulkInserter::~MSSQLBulkInserter() {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Cleaning up database resources" << std::endl;

    if (stmt_ != SQL_NULL_HSTMT) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Releasing ODBC statement handle" << std::endl;
        SQLFreeHandle(SQL_HANDLE_STMT, stmt_);
        stmt_ = SQL_NULL_HSTMT;
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   ODBC statement handle released" << std::endl;
    }

    if (dbc_ != SQL_NULL_HDBC) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Disconnecting from database" << std::endl;
        SQLDisconnect(dbc_);
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Releasing ODBC connection handle" << std::endl;
        SQLFreeHandle(SQL_HANDLE_DBC, dbc_);
        dbc_ = SQL_NULL_HDBC;
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   ODBC connection handle released" << std::endl;
    }

    if (env_ != SQL_NULL_HENV) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Releasing ODBC environment handle" << std::endl;
        SQLFreeHandle(SQL_HANDLE_ENV, env_);
        env_ = SQL_NULL_HENV;
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   ODBC environment handle released" << std::endl;
    }

    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Database resources cleaned up successfully" << std::endl;
}

bool MSSQLBulkInserter::isConnected() const {
    if (dbc_ == SQL_NULL_HDBC) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Connection check - DBC handle is null" << std::endl;
        return false;
    }
    
    SQLINTEGER dead = SQL_CD_TRUE;
    SQLRETURN ret = SQLGetConnectAttr(
        dbc_, SQL_ATTR_CONNECTION_DEAD, &dead, 0, nullptr);
    
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to get connection status" << std::endl;
        return false;
    }
    
    bool connected = dead == SQL_CD_FALSE;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Connection status - "
              << (connected ? "CONNECTED" : "DISCONNECTED") << std::endl;
    return connected;
}

void MSSQLBulkInserter::resetStatement() {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Resetting statement handle" << std::endl;
    
    if (stmt_ != SQL_NULL_HSTMT) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Freeing existing statement handle" << std::endl;
        SQLFreeHandle(SQL_HANDLE_STMT, stmt_);
        stmt_ = SQL_NULL_HSTMT;
    }
    
    SQLRETURN ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc_, &stmt_);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to allocate new statement handle" << std::endl;
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLAllocHandle STMT");
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Statement handle reset successfully" << std::endl;
}

void MSSQLBulkInserter::setAutoCommit(bool enabled) {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Setting auto-commit to "
              << (enabled ? "ON" : "OFF") << std::endl;
    
    SQLRETURN ret = SQLSetConnectAttr(
        dbc_, SQL_ATTR_AUTOCOMMIT,
        enabled ? (SQLPOINTER)SQL_AUTOCOMMIT_ON
                : (SQLPOINTER)SQL_AUTOCOMMIT_OFF,
        SQL_IS_UINTEGER);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to set auto-commit mode" << std::endl;
        throwODBCError(SQL_HANDLE_DBC, dbc_,
                       "SQLSetConnectAttr AUTOCOMMIT");
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Auto-commit mode set successfully" << std::endl;
}

void MSSQLBulkInserter::beginTransaction() {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Beginning database transaction" << std::endl;
    setAutoCommit(false);
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Transaction started successfully" << std::endl;
}

void MSSQLBulkInserter::commitTransaction() {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Committing database transaction" << std::endl;
    
    SQLRETURN ret = SQLEndTran(SQL_HANDLE_DBC, dbc_, SQL_COMMIT);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to commit transaction" << std::endl;
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLEndTran COMMIT");
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Transaction committed successfully" << std::endl;
}

void MSSQLBulkInserter::rollbackTransaction() {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Rolling back database transaction" << std::endl;
    
    SQLRETURN ret = SQLEndTran(SQL_HANDLE_DBC, dbc_, SQL_ROLLBACK);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to rollback transaction" << std::endl;
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLEndTran ROLLBACK");
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Transaction rolled back successfully" << std::endl;
}

void MSSQLBulkInserter::throwODBCError(SQLSMALLINT handle_type,
                                       SQLHANDLE handle,
                                       const std::string& context) const {
    SQLCHAR     sqlstate[6]{};
    SQLCHAR     message[SQL_MAX_MESSAGE_LENGTH]{};
    SQLINTEGER  native_error{};
    SQLSMALLINT length{};

    SQLGetDiagRec(handle_type, handle, 1,
                  sqlstate, &native_error,
                  message, sizeof(message), &length);

    std::string msg = context + ": [" +
                      std::string((char*)sqlstate) + "] " +
                      std::string((char*)message);
    throw std::runtime_error(msg);
}

// ----------------- TRADES BULK INSERT -------------------

void MSSQLBulkInserter::bulkInsertTrades(
    const std::vector<MarketData::Trade>& trades,
    std::size_t batch_size) {

    if (trades.empty()) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: bulkInsertTrades called with empty trade list" << std::endl;
        return;
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Starting bulk insert of " << trades.size() << " trades" << std::endl;
    
    if (!isConnected()) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Cannot insert trades - database not connected" << std::endl;
        throw std::runtime_error("bulkInsertTrades: not connected");
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    
    resetStatement();
    setAutoCommit(false);

    SQLRETURN ret;

    const char* insert_sql =
        "INSERT INTO [dbo].[trades] ("
        "  [timestamp], [exchange], [symbol], [market_type],"
        "  [trade_id], [price], [quantity], [side], [is_buyer_maker]) "
        "VALUES (?,?,?,?,?,?,?,?,?);";

    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Preparing SQL statement for trade insertion" << std::endl;
    ret = SQLPrepare(stmt_, (SQLCHAR*)insert_sql, SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Failed to prepare SQL statement for trades" << std::endl;
        throwODBCError(SQL_HANDLE_STMT, stmt_,
                       "Prepare bulkInsertTrades");
    }
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   SQL statement prepared successfully" << std::endl;

    const std::size_t total = trades.size();
    const std::size_t bs    = batch_size == 0 ? total : batch_size;

    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Total trades: " << total << ", Batch size: " << bs << std::endl;

    constexpr std::size_t EXCH_LEN  = 50;
    constexpr std::size_t SYM_LEN   = 50;
    constexpr std::size_t MTYPE_LEN = 20;
    constexpr std::size_t TID_LEN   = 200;   // generous trade id
    constexpr std::size_t SIDE_LEN  = 10;

    SQLSetStmtAttr(stmt_, SQL_ATTR_PARAM_BIND_TYPE,
                   (SQLPOINTER)SQL_PARAM_BIND_BY_COLUMN, 0);

    std::size_t total_inserted = 0;
    
    for (std::size_t offset = 0; offset < total; ) {
        const std::size_t n = std::min(bs, total - offset);
        
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Processing batch " << (offset/bs + 1) << " with " << n << " trades" << std::endl;

        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAMSET_SIZE,
                       (SQLPOINTER)(SQLULEN)n, 0);

        std::vector<std::array<SQLCHAR, kTimestampStringLength>> ts(n);
        std::vector<std::array<SQLCHAR, EXCH_LEN + 1>>  exch(n);
        std::vector<std::array<SQLCHAR, SYM_LEN + 1>>   sym(n);
        std::vector<std::array<SQLCHAR, MTYPE_LEN + 1>> mtype(n);
        std::vector<std::array<SQLCHAR, TID_LEN + 1>>   tid(n);
        std::vector<double> price(n);
        std::vector<double> qty(n);
        std::vector<std::array<SQLCHAR, SIDE_LEN + 1>>  side(n);
        std::vector<SQLCHAR> buyer_maker(n);

        std::vector<SQLLEN> ind_ts(n), ind_exch(n), ind_sym(n),
                            ind_mtype(n), ind_tid(n),
                            ind_price(n), ind_qty(n),
                            ind_side(n), ind_bm(n);

        for (std::size_t i = 0; i < n; ++i) {
            const auto& t = trades[offset + i];

            copyStrFixed(formatTimestampMicros(t.timestamp_us), ts[i]);

            copyStrFixed(t.exchange,    exch[i]);
            copyStrFixed(t.symbol,      sym[i]);
            copyStrFixed(t.market_type, mtype[i]);
            copyStrFixed(t.trade_id,    tid[i]);
            copyStrFixed(t.side,        side[i]);

            price[i]       = t.price;
            qty[i]         = t.quantity;
            buyer_maker[i] = t.is_buyer_maker ? (SQLCHAR)1 : (SQLCHAR)0;

            ind_ts[i]    = SQL_NTS;
            ind_exch[i]  = SQL_NTS;
            ind_sym[i]   = SQL_NTS;
            ind_mtype[i] = SQL_NTS;
            ind_tid[i]   = SQL_NTS;
            ind_price[i] = 0;
            ind_qty[i]   = 0;
            ind_side[i]  = SQL_NTS;
            ind_bm[i]    = 0;
        }

        // 1: timestamp → DATETIME2(6)
        ret = SQLBindParameter(
            stmt_, 1, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_TYPE_TIMESTAMP,
            kTimestampColumnSize, kTimestampScaleMicros,
            ts[0].data(), kTimestampStringLength,
            ind_ts.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades ts");

        // 2: exchange
        ret = SQLBindParameter(
            stmt_, 2, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            EXCH_LEN, 0,
            exch[0].data(), EXCH_LEN + 1,
            ind_exch.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades exch");

        // 3: symbol
        ret = SQLBindParameter(
            stmt_, 3, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            SYM_LEN, 0,
            sym[0].data(), SYM_LEN + 1,
            ind_sym.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades sym");

        // 4: market_type
        ret = SQLBindParameter(
            stmt_, 4, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            MTYPE_LEN, 0,
            mtype[0].data(), MTYPE_LEN + 1,
            ind_mtype.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "Bind trades market_type");

        // 5: trade_id
        ret = SQLBindParameter(
            stmt_, 5, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            TID_LEN, 0,
            tid[0].data(), TID_LEN + 1,
            ind_tid.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "Bind trades trade_id");

        // 6: price (DECIMAL(20,8))
        ret = SQLBindParameter(
            stmt_, 6, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            20, 8, price.data(), 0, ind_price.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades price");

        // 7: quantity (DECIMAL(30,8))
        ret = SQLBindParameter(
            stmt_, 7, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            30, 8, qty.data(), 0, ind_qty.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades qty");

        // 8: side
        ret = SQLBindParameter(
            stmt_, 8, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            SIDE_LEN, 0,
            side[0].data(), SIDE_LEN + 1,
            ind_side.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades side");

        // 9: is_buyer_maker
        ret = SQLBindParameter(
            stmt_, 9, SQL_PARAM_INPUT,
            SQL_C_BIT, SQL_BIT,
            0, 0, buyer_maker.data(), 0, ind_bm.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades bm");

        ret = SQLExecute(stmt_);
        if (!SQL_SUCCEEDED(ret)) {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Failed to execute trade batch insertion" << std::endl;
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "SQLExecute bulkInsertTrades");
        }

        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Successfully executed batch with " << n << " trades" << std::endl;
        
        SQLFreeStmt(stmt_, SQL_RESET_PARAMS);
        offset += n;
        total_inserted += n;
    }

    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Committing transaction for " << total_inserted << " trades" << std::endl;
    commitTransaction();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Successfully inserted " << total_inserted
              << " trades in " << duration.count() << "ms" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Insertion rate: "
              << (total_inserted * 1000.0 / duration.count()) << " trades/second" << std::endl;
}

// ----------------- OHLCV BULK INSERT -------------------

void MSSQLBulkInserter::ensureKlinesTable(const std::string& table_name) {
    std::lock_guard<std::mutex> lock(ddl_mutex_);

    if (known_klines_tables_.count(table_name)) return;

    resetStatement();

    // Use unnamed UNIQUE constraint to avoid UQ_* name collisions.
    std::string sql =
        "IF OBJECT_ID('dbo." + table_name + "', 'U') IS NULL "
        "BEGIN "
        "CREATE TABLE [dbo].[" + table_name + "] ("
        "  id BIGINT IDENTITY(1,1) PRIMARY KEY,"
        "  [timestamp] DATETIME2(6) NOT NULL,"
        "  exchange VARCHAR(50) NOT NULL,"
        "  symbol VARCHAR(50) NOT NULL,"
        "  market_type VARCHAR(20) NOT NULL DEFAULT 'spot',"
        "  timeframe VARCHAR(10) NOT NULL,"
        "  [open] DECIMAL(20,8) NOT NULL,"
        "  high DECIMAL(20,8) NOT NULL,"
        "  low DECIMAL(20,8) NOT NULL,"
        "  [close] DECIMAL(20,8) NOT NULL,"
        "  volume DECIMAL(30,8) NOT NULL,"
        "  created_at DATETIME2(6) DEFAULT SYSUTCDATETIME(),"
        "  UNIQUE ([timestamp], exchange, symbol, market_type, timeframe)"
        ");"
        "END;";

    SQLRETURN ret = SQLExecDirect(
        stmt_, (SQLCHAR*)sql.c_str(), SQL_NTS);

    if (!SQL_SUCCEEDED(ret)) {
        // swallow "object already exists" just in case
        SQLCHAR     state[6]{};
        SQLINTEGER  native{};
        SQLCHAR     msg[SQL_MAX_MESSAGE_LENGTH]{};
        SQLSMALLINT len{};
        SQLGetDiagRec(SQL_HANDLE_STMT, stmt_, 1,
                      state, &native, msg, sizeof(msg), &len);
        if (std::string((char*)state) != "42S01") {
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "ensureKlinesTable EXEC");
        }
    } else {
        known_klines_tables_.insert(table_name);
    }

    SQLFreeStmt(stmt_, SQL_CLOSE);
}

void MSSQLBulkInserter::bulkInsertOHLCV(
    const std::string& table_name,
    const std::vector<OHLCV>& candles,
    std::size_t batch_size) {

    if (candles.empty()) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: bulkInsertOHLCV called with empty candle list" << std::endl;
        return;
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Starting bulk insert of " << candles.size()
              << " OHLCV candles into table " << table_name << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    ensureKlinesTable(table_name);

    std::string query =
        "INSERT INTO [dbo].[" + table_name + "] "
        "([timestamp], exchange, symbol, market_type, timeframe,"
        " [open], high, low, [close], volume)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)";

    resetStatement();
    SQLRETURN ret = SQLPrepare(
        stmt_, (SQLCHAR*)query.c_str(), SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_STMT, stmt_,
                       "SQLPrepare OHLCV");
    }

    const std::size_t total = candles.size();
    const std::size_t bs    = batch_size == 0 ? total : batch_size;

    SQLSetStmtAttr(stmt_, SQL_ATTR_PARAM_BIND_TYPE,
                   (SQLPOINTER)SQL_PARAM_BIND_BY_COLUMN, 0);

    constexpr std::size_t STR_LEN = 50;

    std::size_t total_inserted = 0;
    
    for (std::size_t offset = 0; offset < total; ) {
        std::size_t n = std::min(bs, total - offset);

        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAMSET_SIZE,
                       (SQLPOINTER)n, 0);

        std::vector<std::array<SQLCHAR, kTimestampStringLength>> ts(n);
        std::vector<std::array<SQLCHAR, STR_LEN + 1>> exch(n), sym(n),
                                                     mtype(n), tf(n);
        std::vector<double> o(n), h(n), l(n), c(n), v(n);

        std::vector<SQLLEN> ind_ts(n), ind_exch(n), ind_sym(n),
                            ind_mtype(n), ind_tf(n),
                            ind_o(n), ind_h(n), ind_l(n),
                            ind_c(n), ind_v(n);

        for (std::size_t i = 0; i < n; ++i) {
            const auto& cd = candles[offset + i];

            copyStrFixed(formatTimestampMicros(cd.timestamp_us), ts[i]);
            ind_ts[i] = SQL_NTS;

            copyStrFixed(cd.exchange,    exch[i]);
            copyStrFixed(cd.symbol,      sym[i]);
            copyStrFixed(cd.market_type, mtype[i]);
            copyStrFixed(cd.timeframe,   tf[i]);

            o[i] = cd.open;
            h[i] = cd.high;
            l[i] = cd.low;
            c[i] = cd.close;
            v[i] = cd.volume;

            ind_exch[i]  = SQL_NTS;
            ind_sym[i]   = SQL_NTS;
            ind_mtype[i] = SQL_NTS;
            ind_tf[i]    = SQL_NTS;
            ind_o[i]     = 0;
            ind_h[i]     = 0;
            ind_l[i]     = 0;
            ind_c[i]     = 0;
            ind_v[i]     = 0;
        }

        // 1: timestamp
        ret = SQLBindParameter(
            stmt_, 1, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_TYPE_TIMESTAMP,
            kTimestampColumnSize, kTimestampScaleMicros,
            ts[0].data(), kTimestampStringLength, ind_ts.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV ts");

        // 2: exchange
        ret = SQLBindParameter(
            stmt_, 2, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            STR_LEN, 0,
            exch[0].data(), STR_LEN + 1,
            ind_exch.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV exch");

        // 3: symbol
        ret = SQLBindParameter(
            stmt_, 3, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            STR_LEN, 0,
            sym[0].data(), STR_LEN + 1,
            ind_sym.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV sym");

        // 4: market_type
        ret = SQLBindParameter(
            stmt_, 4, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            STR_LEN, 0,
            mtype[0].data(), STR_LEN + 1,
            ind_mtype.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV mkt");

        // 5: timeframe
        ret = SQLBindParameter(
            stmt_, 5, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            10, 0,
            tf[0].data(), STR_LEN + 1,
            ind_tf.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV tf");

        // 6..10: OHLCV
        ret = SQLBindParameter(
            stmt_, 6, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            20, 8, o.data(), 0, ind_o.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV open");

        ret = SQLBindParameter(
            stmt_, 7, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            20, 8, h.data(), 0, ind_h.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV high");

        ret = SQLBindParameter(
            stmt_, 8, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            20, 8, l.data(), 0, ind_l.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV low");

        ret = SQLBindParameter(
            stmt_, 9, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            20, 8, c.data(), 0, ind_c.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV close");

        ret = SQLBindParameter(
            stmt_, 10, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            30, 8, v.data(), 0, ind_v.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV vol");

        ret = SQLExecute(stmt_);
        if (!SQL_SUCCEEDED(ret)) {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Failed to execute OHLCV batch insertion" << std::endl;
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "SQLExecute bulkInsertOHLCV");
        }

        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Successfully executed batch with " << n << " candles" << std::endl;
        
        SQLFreeStmt(stmt_, SQL_RESET_PARAMS);
        offset += n;
        total_inserted += n;
    }

    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Committing transaction for " << total_inserted << " candles" << std::endl;
    commitTransaction();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Successfully inserted " << total_inserted
              << " OHLCV candles into " << table_name << " in " << duration.count() << "ms" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Insertion rate: "
              << (total_inserted * 1000.0 / duration.count()) << " candles/second" << std::endl;
}

// ----------------- ORDERBOOK BULK INSERT -------------------

void MSSQLBulkInserter::bulkInsertOrderbooks(
    const std::vector<MarketData::OrderbookSnapshot>& obs,
    std::size_t batch_size) {

    if (obs.empty()) {
        std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: bulkInsertOrderbooks called with empty orderbook list" << std::endl;
        return;
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Starting bulk insert of " << obs.size()
              << " orderbook snapshots" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    ensureCoreTables();

    const char* sql =
        "INSERT INTO [dbo].[orderbook_snapshots] "
        "([timestamp], [exchange], [symbol], [market_type],"
        " [bids], [asks], [checksum])"
        " VALUES (?,?,?,?,?,?,?);";

    constexpr std::size_t EXCH_LEN   = 50;
    constexpr std::size_t SYMBOL_LEN = 50;
    constexpr std::size_t MTYPE_LEN  = 20;
    constexpr std::size_t JSON_LEN   = 2048;
    constexpr std::size_t CK_LEN     = 128;

    std::size_t total_inserted = 0;
    
    for (std::size_t offset = 0; offset < obs.size(); ) {
        std::size_t n = std::min(batch_size, obs.size() - offset);

        resetStatement();

        SQLRETURN ret = SQLPrepare(
            stmt_, (SQLCHAR*)sql, SQL_NTS);
        if (!SQL_SUCCEEDED(ret)) {
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "Prepare bulkInsertOrderbooks");
        }

        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAMSET_SIZE,
                       (SQLPOINTER)n, 0);
        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAM_BIND_TYPE,
                       (SQLPOINTER)SQL_PARAM_BIND_BY_COLUMN, 0);

        std::vector<std::array<SQLCHAR, kTimestampStringLength>> ts(n);
        std::vector<std::array<SQLCHAR, EXCH_LEN + 1>>   exch(n);
        std::vector<std::array<SQLCHAR, SYMBOL_LEN + 1>> sym(n);
        std::vector<std::array<SQLCHAR, MTYPE_LEN + 1>>  mtype(n);
        std::vector<std::array<SQLCHAR, JSON_LEN + 1>>   bids(n);
        std::vector<std::array<SQLCHAR, JSON_LEN + 1>>   asks(n);
        std::vector<std::array<SQLCHAR, CK_LEN + 1>>     ck(n);

        std::vector<SQLLEN> ind_ts(n),
                            ind_exch(n), ind_sym(n), ind_mtype(n),
                            ind_bids(n), ind_asks(n), ind_ck(n);

        auto copyStr = [](const std::string& s, auto& buf) {
            const std::size_t max = buf.size() - 1;
            const std::size_t len = std::min<std::size_t>(max, s.size());
            if (len) {
                std::memcpy(buf.data(), s.data(), len);
            }
            buf[len] = 0;
        };

        for (std::size_t i = 0; i < n; ++i) {
            const auto& ob = obs[offset + i];

            copyStr(formatTimestampMicros(ob.timestamp_us), ts[i]);
            ind_ts[i] = SQL_NTS;

            copyStr(ob.exchange,    exch[i]);
            copyStr(ob.symbol,      sym[i]);
            copyStr(ob.market_type, mtype[i]);
            copyStr(ob.bids_json,   bids[i]);
            copyStr(ob.asks_json,   asks[i]);

            if (!ob.checksum.empty()) {
                copyStr(ob.checksum, ck[i]);
                ind_ck[i] = SQL_NTS;
            } else {
                ck[i][0]  = 0;
                ind_ck[i] = SQL_NULL_DATA;
            }

            ind_exch[i]  = SQL_NTS;
            ind_sym[i]   = SQL_NTS;
            ind_mtype[i] = SQL_NTS;
            ind_bids[i]  = SQL_NTS;
            ind_asks[i]  = SQL_NTS;
        }

        // 1: timestamp
        ret = SQLBindParameter(
            stmt_, 1, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_TYPE_TIMESTAMP,
            kTimestampColumnSize, kTimestampScaleMicros,
            ts[0].data(), kTimestampStringLength,
            ind_ts.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind ob ts");

        // 2: exchange
        ret = SQLBindParameter(
            stmt_, 2, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            EXCH_LEN, 0,
            exch[0].data(), EXCH_LEN + 1,
            ind_exch.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind ob exch");

        // 3: symbol
        ret = SQLBindParameter(
            stmt_, 3, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            SYMBOL_LEN, 0,
            sym[0].data(), SYMBOL_LEN + 1,
            ind_sym.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind ob sym");

        // 4: market_type
        ret = SQLBindParameter(
            stmt_, 4, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            MTYPE_LEN, 0,
            mtype[0].data(), MTYPE_LEN + 1,
            ind_mtype.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind ob mtype");

        // 5: bids JSON
        ret = SQLBindParameter(
            stmt_, 5, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            JSON_LEN, 0,
            bids[0].data(), JSON_LEN + 1,
            ind_bids.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind ob bids");

        // 6: asks JSON
        ret = SQLBindParameter(
            stmt_, 6, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            JSON_LEN, 0,
            asks[0].data(), JSON_LEN + 1,
            ind_asks.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind ob asks");

        // 7: checksum
        ret = SQLBindParameter(
            stmt_, 7, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            CK_LEN, 0,
            ck[0].data(), CK_LEN + 1,
            ind_ck.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind ob checksum");

        ret = SQLExecute(stmt_);
        if (!SQL_SUCCEEDED(ret)) {
            std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Failed to execute orderbook batch insertion" << std::endl;
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "SQLExecute bulkInsertOrderbooks");
        }

        std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Successfully executed batch with " << n << " orderbooks" << std::endl;
        
        SQLFreeStmt(stmt_, SQL_RESET_PARAMS);
        offset += n;
        total_inserted += n;
    }

    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Committing transaction for " << total_inserted << " orderbooks" << std::endl;
    commitTransaction();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Successfully inserted " << total_inserted
              << " orderbook snapshots in " << duration.count() << "ms" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG]   Insertion rate: "
              << (total_inserted * 1000.0 / duration.count()) << " orderbooks/second" << std::endl;
}

// ----------------- CORE TABLES & IDENT HELPERS -------------------

void MSSQLBulkInserter::ensureCoreTables() {
    std::lock_guard<std::mutex> lock(ddl_mutex_);

    resetStatement();

    auto exec = [&](const std::string& sql, const char* ctx) {
        SQLRETURN r = SQLExecDirect(
            stmt_, (SQLCHAR*)sql.c_str(), SQL_NTS);
        if (!SQL_SUCCEEDED(r)) {
            throwODBCError(SQL_HANDLE_STMT, stmt_, ctx);
        }
        SQLFreeStmt(stmt_, SQL_CLOSE);
    };

    const std::string trades_sql =
        "IF OBJECT_ID('dbo.trades', 'U') IS NULL "
        "BEGIN "
        "CREATE TABLE [dbo].[trades] ("
        "  id BIGINT IDENTITY(1,1) PRIMARY KEY,"
        "  [timestamp] DATETIME2(6) NOT NULL,"
        "  exchange VARCHAR(50) NOT NULL,"
        "  symbol VARCHAR(50) NOT NULL,"
        "  market_type VARCHAR(20) NOT NULL DEFAULT 'spot',"
        "  trade_id VARCHAR(200),"
        "  price DECIMAL(20,8) NOT NULL,"
        "  quantity DECIMAL(30,8) NOT NULL,"
        "  side VARCHAR(10) NOT NULL,"
        "  is_buyer_maker BIT,"
        "  created_at DATETIME2(6) DEFAULT SYSUTCDATETIME()"
        ");"
        "CREATE INDEX idx_trades_lookup "
        "ON [dbo].[trades](exchange, symbol, market_type, [timestamp] DESC);"
        "END;";

    const std::string ob_sql =
        "IF OBJECT_ID('dbo.orderbook_snapshots', 'U') IS NULL "
        "BEGIN "
        "CREATE TABLE [dbo].[orderbook_snapshots] ("
        "  id BIGINT IDENTITY(1,1) PRIMARY KEY,"
        "  [timestamp] DATETIME2(6) NOT NULL,"
        "  exchange VARCHAR(50) NOT NULL,"
        "  symbol VARCHAR(50) NOT NULL,"
        "  market_type VARCHAR(20) NOT NULL DEFAULT 'spot',"
        "  bids NVARCHAR(MAX) NOT NULL,"
        "  asks NVARCHAR(MAX) NOT NULL,"
        "  checksum VARCHAR(64),"
        "  created_at DATETIME2(6) DEFAULT SYSUTCDATETIME()"
        ");"
        "CREATE INDEX idx_orderbook_lookup "
        "ON [dbo].[orderbook_snapshots](exchange, symbol, market_type, [timestamp] DESC);"
        "END;";

    exec(trades_sql, "ensureCoreTables trades");
    exec(ob_sql,     "ensureCoreTables orderbook_snapshots");
}

std::string MSSQLBulkInserter::sanitizeIdentifier(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) ||
            c == '_' || c == '-') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        throw std::runtime_error("sanitizeIdentifier: empty name");
    }
    return out;
}

bool MSSQLBulkInserter::verifyConnection() const {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Verifying database connection..." << std::endl;
    
    if (!isConnected()) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Connection check failed - not connected" << std::endl;
        return false;
    }
    
    // Test with a simple query
    SQLHSTMT test_stmt;
    SQLRETURN ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc_, &test_stmt);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to allocate test statement handle" << std::endl;
        return false;
    }
    
    ret = SQLExecDirect(test_stmt, (SQLCHAR*)"SELECT 1", SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Test query failed" << std::endl;
        SQLFreeHandle(SQL_HANDLE_STMT, test_stmt);
        return false;
    }
    
    SQLINTEGER value;
    ret = SQLBindCol(test_stmt, 1, SQL_C_LONG, &value, 0, nullptr);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to bind test result" << std::endl;
        SQLFreeHandle(SQL_HANDLE_STMT, test_stmt);
        return false;
    }
    
    ret = SQLFetch(test_stmt);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to fetch test result" << std::endl;
        SQLFreeHandle(SQL_HANDLE_STMT, test_stmt);
        return false;
    }
    
    if (value != 1) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Unexpected test result: " << value << std::endl;
        SQLFreeHandle(SQL_HANDLE_STMT, test_stmt);
        return false;
    }
    
    SQLFreeHandle(SQL_HANDLE_STMT, test_stmt);
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Connection verification successful (test query returned 1)" << std::endl;
    return true;
}

void MSSQLBulkInserter::testTableCreation() {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MSSQLBulkInserter: Testing table creation..." << std::endl;
    
    resetStatement();
    
    // Create a test table
    std::string test_sql = 
        "IF OBJECT_ID('dbo.connection_test', 'U') IS NULL "
        "BEGIN "
        "CREATE TABLE [dbo].[connection_test] ("
        "  id INT IDENTITY(1,1) PRIMARY KEY," 
        "  test_value VARCHAR(100),"
        "  created_at DATETIME2 DEFAULT GETDATE()"
        ");"
        "END;";
    
    SQLRETURN ret = SQLExecDirect(stmt_, (SQLCHAR*)test_sql.c_str(), SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to create test table" << std::endl;
        throwODBCError(SQL_HANDLE_STMT, stmt_, "Test table creation");
    }
    
    // Insert a test record
    SQLBindParameter(stmt_, 1, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 100, 0, 
                     (SQLCHAR*)"Connection test successful", SQL_NTS, nullptr);
    
    ret = SQLExecDirect(stmt_, (SQLCHAR*)"INSERT INTO [dbo].[connection_test] (test_value) VALUES (?)", SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to insert test record" << std::endl;
        throwODBCError(SQL_HANDLE_STMT, stmt_, "Test record insertion");
    }
    
    // Query the test record back
    ret = SQLExecDirect(stmt_, (SQLCHAR*)"SELECT test_value FROM [dbo].[connection_test] WHERE id = (SELECT MAX(id) FROM [dbo].[connection_test])", SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MSSQLBulkInserter: Failed to query test record" << std::endl;
        throwODBCError(SQL_HANDLE_STMT, stmt_, "Test record query");
    }
    
    char buffer[101];
    SQLLEN indicator;
    SQLBindCol(stmt_, 1, SQL_C_CHAR, buffer, sizeof(buffer), &indicator);
    ret = SQLFetch(stmt_);
    if (SQL_SUCCEEDED(ret)) {
        std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Test record retrieved: " << buffer << std::endl;
    }
    
    // Clean up
    SQLFreeStmt(stmt_, SQL_CLOSE);
    SQLExecDirect(stmt_, (SQLCHAR*)"DROP TABLE [dbo].[connection_test]", SQL_NTS);
    
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MSSQLBulkInserter: Table creation test completed successfully" << std::endl;
}
