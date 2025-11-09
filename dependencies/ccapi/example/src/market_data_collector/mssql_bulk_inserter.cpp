#include "mssql_bulk_inserter.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <array>
#include <sql.h>

using namespace MarketData;

MSSQLBulkInserter::MSSQLBulkInserter(const std::string& connection_string)
    : connection_string_(connection_string) {
    SQLRETURN ret;
    ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env_);
    ensureCoreTables();
    if (!SQL_SUCCEEDED(ret)) {
        throw std::runtime_error("Failed to allocate ODBC environment");
    }

    ret = SQLSetEnvAttr(env_, SQL_ATTR_ODBC_VERSION,
                        reinterpret_cast<SQLPOINTER>(SQL_OV_ODBC3), 0);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_ENV, env_, "SQLSetEnvAttr");
    }

    ret = SQLAllocHandle(SQL_HANDLE_DBC, env_, &dbc_);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_ENV, env_, "SQLAllocHandle DBC");
    }

    ret = SQLDriverConnect(
        dbc_, nullptr,
        (SQLCHAR*)connection_string_.c_str(), SQL_NTS,
        nullptr, 0, nullptr, SQL_DRIVER_NOPROMPT);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLDriverConnect");
    }

    ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc_, &stmt_);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLAllocHandle STMT");
    }

    setAutoCommit(false);
}

MSSQLBulkInserter::~MSSQLBulkInserter() {
    if (stmt_ != SQL_NULL_HSTMT) {
        SQLFreeHandle(SQL_HANDLE_STMT, stmt_);
        stmt_ = SQL_NULL_HSTMT;
    }
    if (dbc_ != SQL_NULL_HDBC) {
        SQLDisconnect(dbc_);
        SQLFreeHandle(SQL_HANDLE_DBC, dbc_);
        dbc_ = SQL_NULL_HDBC;
    }
    if (env_ != SQL_NULL_HENV) {
        SQLFreeHandle(SQL_HANDLE_ENV, env_);
        env_ = SQL_NULL_HENV;
    }
}

bool MSSQLBulkInserter::isConnected() const {
    if (dbc_ == SQL_NULL_HDBC) return false;
    SQLINTEGER dead = SQL_CD_TRUE;
    SQLRETURN ret = SQLGetConnectAttr(
        dbc_, SQL_ATTR_CONNECTION_DEAD, &dead, 0, nullptr);
    return SQL_SUCCEEDED(ret) && dead == SQL_CD_FALSE;
}

void MSSQLBulkInserter::resetStatement() {
    if (stmt_ != SQL_NULL_HSTMT) {
        SQLFreeHandle(SQL_HANDLE_STMT, stmt_);
        stmt_ = SQL_NULL_HSTMT;
    }
    SQLRETURN ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc_, &stmt_);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLAllocHandle STMT");
    }
}

void MSSQLBulkInserter::setAutoCommit(bool enabled) {
    SQLRETURN ret = SQLSetConnectAttr(
        dbc_, SQL_ATTR_AUTOCOMMIT,
        enabled ? (SQLPOINTER)SQL_AUTOCOMMIT_ON
                : (SQLPOINTER)SQL_AUTOCOMMIT_OFF,
        SQL_IS_UINTEGER);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLSetConnectAttr AUTOCOMMIT");
    }
}

void MSSQLBulkInserter::beginTransaction() {
    setAutoCommit(false);
}

void MSSQLBulkInserter::commitTransaction() {
    SQLEndTran(SQL_HANDLE_DBC, dbc_, SQL_COMMIT);
}

void MSSQLBulkInserter::rollbackTransaction() {
    SQLEndTran(SQL_HANDLE_DBC, dbc_, SQL_ROLLBACK);
}

SQL_TIMESTAMP_STRUCT MSSQLBulkInserter::toSqlTimestamp(int64_t timestamp_ms) {
    using namespace std::chrono;
    system_clock::time_point tp{milliseconds(timestamp_ms)};
    std::time_t tt = system_clock::to_time_t(tp);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    SQL_TIMESTAMP_STRUCT ts{};
    ts.year   = tm.tm_year + 1900;
    ts.month  = tm.tm_mon + 1;
    ts.day    = tm.tm_mday;
    ts.hour   = tm.tm_hour;
    ts.minute = tm.tm_min;
    ts.second = tm.tm_sec;
    ts.fraction = 0; // no fractional seconds -> no scale mismatch
    return ts;
}

void MSSQLBulkInserter::throwODBCError(SQLSMALLINT handle_type,
                                       SQLHANDLE handle,
                                       const std::string& context) const {
    SQLCHAR sqlstate[6]{};
    SQLCHAR message[SQL_MAX_MESSAGE_LENGTH]{};
    SQLINTEGER native_error{};
    SQLSMALLINT length{};
    SQLGetDiagRec(handle_type, handle, 1,
                  sqlstate, &native_error,
                  message, sizeof(message), &length);
    std::string msg =
        context + ": [" + std::string((char*)sqlstate) + "] " +
        std::string((char*)message);
    throw std::runtime_error(msg);
}

// ----------------- TRADES BULK INSERT -------------------

void MSSQLBulkInserter::bulkInsertTrades(
    const std::vector<Trade>& trades,
    std::size_t batch_size) {

    if (trades.empty()) return;

    const std::string query =
        "INSERT INTO trades("
        "timestamp, exchange, symbol, market_type, trade_id, "
        "price, quantity, side, is_buyer_maker"
        ") VALUES (?,?,?,?,?,?,?,?,?)";

    resetStatement();
    SQLRETURN ret = SQLPrepare(stmt_,
                               (SQLCHAR*)query.c_str(),
                               SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_STMT, stmt_, "SQLPrepare trades");
    }

    std::size_t offset = 0;
    while (offset < trades.size()) {
        std::size_t n = std::min(batch_size, trades.size() - offset);

        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAMSET_SIZE,
                       (SQLPOINTER)n, 0);

        std::vector<SQL_TIMESTAMP_STRUCT> ts(n);
        std::vector<std::array<SQLCHAR, 51>> exch(n), sym(n),
                                            mtype(n), tid(n),
                                            side(n);
        std::vector<double> price(n), qty(n);
        std::vector<SQLCHAR> buyer_maker(n);

        std::vector<SQLLEN> ind_ts(n), ind_exch(n), ind_sym(n),
                            ind_mtype(n), ind_tid(n),
                            ind_price(n), ind_qty(n),
                            ind_side(n), ind_bm(n);

        for (std::size_t i = 0; i < n; ++i) {
            const auto& t = trades[offset + i];
            ts[i] = toSqlTimestamp(t.timestamp_ms);

            auto copyStr = [](const std::string& s,
                              std::array<SQLCHAR, 51>& buf) {
                std::size_t len = std::min<std::size_t>(s.size(), 50);
                std::memcpy(buf.data(), s.data(), len);
                buf[len] = 0;
            };

            copyStr(t.exchange, exch[i]);
            copyStr(t.symbol, sym[i]);
            copyStr(t.market_type, mtype[i]);
            copyStr(t.trade_id, tid[i]);
            copyStr(t.side, side[i]);

            price[i] = t.price;
            qty[i]   = t.quantity;
            buyer_maker[i] = t.is_buyer_maker ? 1 : 0;

            ind_ts[i]    = sizeof(SQL_TIMESTAMP_STRUCT);
            ind_exch[i]  = SQL_NTS;
            ind_sym[i]   = SQL_NTS;
            ind_mtype[i] = SQL_NTS;
            ind_tid[i]   = SQL_NTS;
            ind_price[i] = 0;
            ind_qty[i]   = 0;
            ind_side[i]  = SQL_NTS;
            ind_bm[i]    = 0;
        }

        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAM_BIND_TYPE,
                       (SQLPOINTER)SQL_PARAM_BIND_BY_COLUMN, 0);

        // 1: timestamp
        ret = SQLBindParameter(
            stmt_, 1, SQL_PARAM_INPUT,
            SQL_C_TYPE_TIMESTAMP, SQL_TYPE_TIMESTAMP,
            0, 0, ts.data(), 0, ind_ts.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades ts");

        // 2: exchange
        ret = SQLBindParameter(
            stmt_, 2, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            50, 0, exch[0].data(), 50, ind_exch.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades exch");

        // 3: symbol
        ret = SQLBindParameter(
            stmt_, 3, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            50, 0, sym[0].data(), 50, ind_sym.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades sym");

        // 4: market_type
        ret = SQLBindParameter(
            stmt_, 4, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            20, 0, mtype[0].data(), 20, ind_mtype.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades mkt");

        // 5: trade_id
        ret = SQLBindParameter(
            stmt_, 5, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            100, 0, tid[0].data(), 100, ind_tid.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades tid");

        // 6: price
        ret = SQLBindParameter(
            stmt_, 6, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            20, 8, price.data(), 0, ind_price.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades price");

        // 7: quantity
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
            10, 0, side[0].data(), 10, ind_side.data());
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
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "SQLExecute bulkInsertTrades");
        }

        SQLFreeStmt(stmt_, SQL_RESET_PARAMS);
        offset += n;
    }

    commitTransaction();
}

// ----------------- OHLCV BULK INSERT -------------------
void MSSQLBulkInserter::ensureKlinesTable(const std::string& table_name) {
    {
        std::lock_guard<std::mutex> lock(ddl_mutex_);
        if (known_klines_tables_.count(table_name)) return;
    }

    std::lock_guard<std::mutex> lock(ddl_mutex_);
    if (known_klines_tables_.count(table_name)) return;

    SQLHSTMT ddl_stmt = nullptr;
    SQLRETURN ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc_, &ddl_stmt);
    if (!SQL_SUCCEEDED(ret) || !ddl_stmt) {
        std::cerr
            << "ensureKlinesTable: SQLAllocHandle failed for table "
            << table_name << ", skipping auto-DDL\n";
        return; // <-- do NOT throw here
    }

    std::string uq = "UQ_" + table_name;
    for (char& c : uq) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
            c = '_';
    }

    std::string sql =
        "IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = '" + table_name + "') "
        "BEGIN "
        "CREATE TABLE [" + table_name + "] ("
        "  id BIGINT IDENTITY(1,1) PRIMARY KEY,"
        "  timestamp DATETIME2 NOT NULL,"
        "  exchange VARCHAR(50) NOT NULL,"
        "  symbol VARCHAR(50) NOT NULL,"
        "  market_type VARCHAR(20) NOT NULL DEFAULT 'spot',"
        "  timeframe VARCHAR(10) NOT NULL,"
        "  [open] DECIMAL(20,8) NOT NULL,"
        "  high DECIMAL(20,8) NOT NULL,"
        "  low DECIMAL(20,8) NOT NULL,"
        "  [close] DECIMAL(20,8) NOT NULL,"
        "  volume DECIMAL(30,8) NOT NULL,"
        "  created_at DATETIME2 DEFAULT SYSUTCDATETIME(),"
        "  CONSTRAINT [" + uq + "] "
        "    UNIQUE(timestamp, exchange, symbol, market_type, timeframe)"
        ");"
        "END;";

    SQLRETURN r = SQLExecDirect( // TODO CRRITCIAL :: need to rework this C-style cast madness everywhere, like something std::string_view, or find more smarties on the way figuring oput
                                 //                :: also figure out why im this plain stupid right now to copy over accept directly,.. mhh
        ddl_stmt,
        reinterpret_cast<SQLCHAR*>(const_cast<char*>(sql.c_str())),
        SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_STMT, ddl_stmt,
                       "ensureKlinesTable EXEC");
    }

    SQLFreeHandle(SQL_HANDLE_STMT, ddl_stmt);
    known_klines_tables_.insert(table_name);
}

void MSSQLBulkInserter::bulkInsertOHLCV(
    const std::string& table_name,
    const std::vector<OHLCV>& candles,
    std::size_t batch_size) {

    if (candles.empty()) return;

    const std::string query =
        "INSERT INTO " + table_name +
        "(timestamp, exchange, symbol, market_type, timeframe, "
        "[open], high, low, [close], volume) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)";

    resetStatement();
    SQLRETURN ret = SQLPrepare(stmt_,
                               (SQLCHAR*)query.c_str(),
                               SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_STMT, stmt_, "SQLPrepare OHLCV");
    }

    std::size_t offset = 0;
    while (offset < candles.size()) {
        std::size_t n = std::min(batch_size, candles.size() - offset);

        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAMSET_SIZE,
                       (SQLPOINTER)n, 0);
        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAM_BIND_TYPE,
                       (SQLPOINTER)SQL_PARAM_BIND_BY_COLUMN, 0);

        std::vector<SQL_TIMESTAMP_STRUCT> ts(n);
        std::vector<std::array<SQLCHAR, 51>> exch(n), sym(n),
                                            mtype(n), tf(n);
        std::vector<double> o(n), h(n), l(n), c(n), v(n);

        std::vector<SQLLEN> ind_ts(n), ind_exch(n), ind_sym(n),
                            ind_mtype(n), ind_tf(n),
                            ind_o(n), ind_h(n), ind_l(n),
                            ind_c(n), ind_v(n);

        auto copyStr = [](const std::string& s,
                          std::array<SQLCHAR, 51>& buf) {
            std::size_t len = std::min<std::size_t>(s.size(), 50);
            std::memcpy(buf.data(), s.data(), len);
            buf[len] = 0;
        };

        for (std::size_t i = 0; i < n; ++i) {
            const auto& cd = candles[offset + i];
            ts[i] = toSqlTimestamp(cd.timestamp_ms);
            copyStr(cd.exchange, exch[i]);
            copyStr(cd.symbol, sym[i]);
            copyStr(cd.market_type, mtype[i]);
            copyStr(cd.timeframe, tf[i]);

            o[i] = cd.open;
            h[i] = cd.high;
            l[i] = cd.low;
            c[i] = cd.close;
            v[i] = cd.volume;

            ind_ts[i]     = sizeof(SQL_TIMESTAMP_STRUCT);
            ind_exch[i]   = SQL_NTS;
            ind_sym[i]    = SQL_NTS;
            ind_mtype[i]  = SQL_NTS;
            ind_tf[i]     = SQL_NTS;
            ind_o[i]      = 0;
            ind_h[i]      = 0;
            ind_l[i]      = 0;
            ind_c[i]      = 0;
            ind_v[i]      = 0;
        }

        // Bind params
        ret = SQLBindParameter(
            stmt_, 1, SQL_PARAM_INPUT,
            SQL_C_TYPE_TIMESTAMP, SQL_TYPE_TIMESTAMP,
            0, 0, ts.data(), 0, ind_ts.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV ts");

        ret = SQLBindParameter(
            stmt_, 2, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            50, 0, exch[0].data(), 50, ind_exch.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV exch");

        ret = SQLBindParameter(
            stmt_, 3, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            50, 0, sym[0].data(), 50, ind_sym.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV sym");

        ret = SQLBindParameter(
            stmt_, 4, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            20, 0, mtype[0].data(), 20, ind_mtype.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV mkt");

        ret = SQLBindParameter(
            stmt_, 5, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            10, 0, tf[0].data(), 10, ind_tf.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OHLCV tf");

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
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "SQLExecute bulkInsertOHLCV");
        }

        SQLFreeStmt(stmt_, SQL_RESET_PARAMS);
        offset += n;
    }

    commitTransaction();
}

// ----------------- ORDERBOOK BULK INSERT -------------------

void MSSQLBulkInserter::bulkInsertOrderbooks(
    const std::vector<OrderbookSnapshot>& snapshots,
    std::size_t batch_size) {

    if (snapshots.empty()) return;

    const std::string query =
        "INSERT INTO orderbook_snapshots("
        "timestamp, exchange, symbol, market_type, "
        "bids, asks, checksum) VALUES (?,?,?,?,?,?,?)";

    resetStatement();
    SQLRETURN ret = SQLPrepare(stmt_,
                               (SQLCHAR*)query.c_str(),
                               SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_STMT, stmt_, "SQLPrepare orderbooks");
    }

    std::size_t offset = 0;
    while (offset < snapshots.size()) {
        std::size_t n = std::min(batch_size, snapshots.size() - offset);

        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAMSET_SIZE,
                       (SQLPOINTER)n, 0);
        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAM_BIND_TYPE,
                       (SQLPOINTER)SQL_PARAM_BIND_BY_COLUMN, 0);

        std::vector<SQL_TIMESTAMP_STRUCT> ts(n);
        std::vector<std::array<SQLCHAR, 51>> exch(n), sym(n), mtype(n);
        std::vector<std::vector<SQLCHAR>> bids(n), asks(n), ck(n);
        // For simplicity, store JSON as VARCHAR (driver will cast to NVARCHAR)

        std::vector<SQLLEN> ind_ts(n), ind_exch(n), ind_sym(n),
                            ind_mtype(n), ind_bids(n),
                            ind_asks(n), ind_ck(n);

        for (std::size_t i = 0; i < n; ++i) {
            const auto& ob = snapshots[offset + i];
            ts[i] = toSqlTimestamp(ob.timestamp_ms);

            auto copyStr50 = [](const std::string& s,
                                std::array<SQLCHAR, 51>& buf) {
                std::size_t len = std::min<std::size_t>(s.size(), 50);
                std::memcpy(buf.data(), s.data(), len);
                buf[len] = 0;
            };

            copyStr50(ob.exchange, exch[i]);
            copyStr50(ob.symbol, sym[i]);
            copyStr50(ob.market_type, mtype[i]);

            bids[i].assign(ob.bids_json.begin(), ob.bids_json.end());
            bids[i].push_back(0);
            asks[i].assign(ob.asks_json.begin(), ob.asks_json.end());
            asks[i].push_back(0);
            ck[i].assign(ob.checksum.begin(), ob.checksum.end());
            ck[i].push_back(0);

            ind_ts[i]     = sizeof(SQL_TIMESTAMP_STRUCT);
            ind_exch[i]   = SQL_NTS;
            ind_sym[i]    = SQL_NTS;
            ind_mtype[i]  = SQL_NTS;
            ind_bids[i]   = SQL_NTS;
            ind_asks[i]   = SQL_NTS;
            ind_ck[i]     = SQL_NTS;
        }

        ret = SQLBindParameter(
            stmt_, 1, SQL_PARAM_INPUT,
            SQL_C_TYPE_TIMESTAMP, SQL_TYPE_TIMESTAMP,
            0, 0, ts.data(), 0, ind_ts.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OB ts");

        ret = SQLBindParameter(
            stmt_, 2, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            50, 0, exch[0].data(), 50, ind_exch.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OB exch");

        ret = SQLBindParameter(
            stmt_, 3, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            50, 0, sym[0].data(), 50, ind_sym.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OB sym");

        ret = SQLBindParameter(
            stmt_, 4, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            20, 0, mtype[0].data(), 20, ind_mtype.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OB mkt");

        ret = SQLBindParameter(
            stmt_, 5, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            0, 0, bids[0].data(), 0, ind_bids.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OB bids");

        ret = SQLBindParameter(
            stmt_, 6, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            0, 0, asks[0].data(), 0, ind_asks.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OB asks");

        ret = SQLBindParameter(
            stmt_, 7, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            64, 0, ck[0].data(), 64, ind_ck.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind OB checksum");

        ret = SQLExecute(stmt_);
        if (!SQL_SUCCEEDED(ret)) {
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "SQLExecute bulkInsertOrderbooks");
        }

        SQLFreeStmt(stmt_, SQL_RESET_PARAMS);
        offset += n;
    }

    commitTransaction();
}

void MSSQLBulkInserter::ensureCoreTables() {
    std::lock_guard<std::mutex> lock(ddl_mutex_);

    SQLHSTMT ddl_stmt = nullptr;
    SQLRETURN ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc_, &ddl_stmt);
    if (!SQL_SUCCEEDED(ret) || !ddl_stmt) {
        std::cerr
            << "ensureCoreTables: SQLAllocHandle failed, skipping auto-DDL\n";
        return; // <-- do NOT throw here
    }

    auto exec = [&](const std::string& sql, const char* ctx) {
        SQLRETURN r = SQLExecDirect(
            ddl_stmt,
        reinterpret_cast<SQLCHAR*>(const_cast<char*>(sql.c_str())),
        SQL_NTS);
        if (!SQL_SUCCEEDED(r)) {
            // still throw here: DDL itself is broken
            throwODBCError(SQL_HANDLE_STMT, ddl_stmt, ctx);
        }
    };

    const std::string trades_sql =
        "IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'trades') "
        "BEGIN "
        "CREATE TABLE [trades] ("
        "  id BIGINT IDENTITY(1,1) PRIMARY KEY,"
        "  timestamp DATETIME2 NOT NULL,"
        "  exchange VARCHAR(50) NOT NULL,"
        "  symbol VARCHAR(50) NOT NULL,"
        "  market_type VARCHAR(20) NOT NULL DEFAULT 'spot',"
        "  trade_id VARCHAR(100),"
        "  price DECIMAL(20,8) NOT NULL,"
        "  quantity DECIMAL(30,8) NOT NULL,"
        "  side VARCHAR(10) NOT NULL,"
        "  is_buyer_maker BIT,"
        "  created_at DATETIME2 DEFAULT SYSUTCDATETIME()"
        ");"
        "CREATE INDEX idx_trades_lookup "
        "ON [trades](exchange, symbol, market_type, timestamp DESC);"
        "END;";

    const std::string ob_sql =
        "IF NOT EXISTS (SELECT 1 FROM sys.tables "
        "               WHERE name = 'orderbook_snapshots') "
        "BEGIN "
        "CREATE TABLE [orderbook_snapshots] ("
        "  id BIGINT IDENTITY(1,1) PRIMARY KEY,"
        "  timestamp DATETIME2 NOT NULL,"
        "  exchange VARCHAR(50) NOT NULL,"
        "  symbol VARCHAR(50) NOT NULL,"
        "  market_type VARCHAR(20) NOT NULL DEFAULT 'spot',"
        "  bids NVARCHAR(MAX) NOT NULL,"
        "  asks NVARCHAR(MAX) NOT NULL,"
        "  checksum VARCHAR(64),"
        "  created_at DATETIME2 DEFAULT SYSUTCDATETIME()"
        ");"
        "CREATE INDEX idx_orderbook_lookup "
        "ON [orderbook_snapshots](exchange, symbol, market_type, timestamp DESC);"
        "END;";

    exec(trades_sql, "ensureCoreTables trades");
    exec(ob_sql,     "ensureCoreTables orderbook_snapshots");

    SQLFreeHandle(SQL_HANDLE_STMT, ddl_stmt);
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