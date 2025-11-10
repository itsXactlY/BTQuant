#include "mssql_bulk_inserter.h"

#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <cctype>
#include <sql.h>
#include <sqlext.h>

using namespace MarketData;

namespace {

// convert UNIX epoch microseconds -> SQL_TIMESTAMP_STRUCT (for DATETIME2(6))
inline SQL_TIMESTAMP_STRUCT makeSqlTsFromEpochUs(int64_t epoch_us) {
    using namespace std::chrono;
    std::time_t sec = static_cast<std::time_t>(epoch_us / 1000000);
    int64_t usec    = epoch_us % 1000000;
    if (usec < 0) {
        usec += 1000000;
        --sec;
    }

    std::tm tm_utc{};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &sec);
#else
    gmtime_r(&sec, &tm_utc);
#endif

    SQL_TIMESTAMP_STRUCT ts{};
    ts.year   = tm_utc.tm_year + 1900;
    ts.month  = tm_utc.tm_mon + 1;
    ts.day    = tm_utc.tm_mday;
    ts.hour   = tm_utc.tm_hour;
    ts.minute = tm_utc.tm_min;
    ts.second = tm_utc.tm_sec;
    ts.fraction = static_cast<SQLUINTEGER>(usec * 1000); // micro -> nano
    return ts;
}

namespace {
template <std::size_t N>
inline void copyStrFixed(const std::string& s, std::array<SQLCHAR, N>& buf) {
    static_assert(N >= 1, "buffer must have at least 1 byte");
    const std::size_t max_len = N - 1;
    const std::size_t len     = s.size() > max_len ? max_len : s.size();
    if (len) std::memcpy(buf.data(), s.data(), len);
    buf[len] = 0;
}
} // namespace

} // anonymous namespace

//
namespace {
inline SQL_TIMESTAMP_STRUCT ts_from_epoch_us(int64_t epoch_us) {
    using namespace std::chrono;
    int64_t sec = epoch_us / 1'000'000;
    int64_t us  = epoch_us % 1'000'000;
    
    // Handle negative microseconds
    if (us < 0) {
        us += 1'000'000;
        --sec;
    }

    std::tm tm_utc{};
#if defined(_WIN32)
    time_t tt = static_cast<time_t>(sec);
    gmtime_s(&tm_utc, &tt);
#else
    time_t tt = static_cast<time_t>(sec);
    gmtime_r(&tt, &tm_utc);
#endif

    SQL_TIMESTAMP_STRUCT ts{};
    ts.year     = tm_utc.tm_year + 1900;
    ts.month    = tm_utc.tm_mon + 1;
    ts.day      = tm_utc.tm_mday;
    ts.hour     = tm_utc.tm_hour;
    ts.minute   = tm_utc.tm_min;
    ts.second   = tm_utc.tm_sec;
    ts.fraction = static_cast<unsigned long>(us) * 1000UL;
    return ts;
}
} // namespace

MSSQLBulkInserter::MSSQLBulkInserter(const std::string& connection_string)
    : connection_string_(connection_string) {
    SQLRETURN ret;

    // ENV
    ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env_);
    if (!SQL_SUCCEEDED(ret)) {
        throw std::runtime_error("SQLAllocHandle ENV");
    }

    ret = SQLSetEnvAttr(env_, SQL_ATTR_ODBC_VERSION,
                        (SQLPOINTER)SQL_OV_ODBC3, 0);
    if (!SQL_SUCCEEDED(ret)) {
        SQLFreeHandle(SQL_HANDLE_ENV, env_);
        throw std::runtime_error("SQLSetEnvAttr ODBC_VERSION");
    }

    // DBC
    ret = SQLAllocHandle(SQL_HANDLE_DBC, env_, &dbc_);
    if (!SQL_SUCCEEDED(ret)) {
        SQLFreeHandle(SQL_HANDLE_ENV, env_);
        throw std::runtime_error("SQLAllocHandle DBC");
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
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLDriverConnect");
    }

    // STMT used for data-path & DDL
    ret = SQLAllocHandle(SQL_HANDLE_STMT, dbc_, &stmt_);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_DBC, dbc_, "SQLAllocHandle STMT");
    }

    setAutoCommit(false);
    ensureCoreTables();
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
        throwODBCError(SQL_HANDLE_DBC, dbc_,
                       "SQLSetConnectAttr AUTOCOMMIT");
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

    if (trades.empty()) return;
    if (!isConnected()) {
        throw std::runtime_error("bulkInsertTrades: not connected");
    }

    resetStatement();
    setAutoCommit(false);

    SQLRETURN ret;

    const char* insert_sql =
        "INSERT INTO [dbo].[trades] ("
        "  [timestamp], [exchange], [symbol], [market_type], [trade_id],"
        "  [price], [quantity], [side], [is_buyer_maker]"
        ") VALUES (?,?,?,?,?,?,?,?,?)";

    ret = SQLPrepare(stmt_, (SQLCHAR*)insert_sql, SQL_NTS);
    if (!SQL_SUCCEEDED(ret)) {
        throwODBCError(SQL_HANDLE_STMT, stmt_,
                       "Prepare bulkInsertTrades");
    }

    const std::size_t total = trades.size();
    const std::size_t bs    = batch_size == 0 ? total : batch_size;

    // column-wise binding
    SQLSetStmtAttr(stmt_, SQL_ATTR_PARAM_BIND_TYPE,
                   (SQLPOINTER)SQL_PARAM_BIND_BY_COLUMN, 0);

    constexpr std::size_t EXCH_LEN  = 50;
    constexpr std::size_t SYM_LEN   = 50;
    constexpr std::size_t MTYPE_LEN = 20;
    constexpr std::size_t TID_LEN   = 200;   // generous trade id
    constexpr std::size_t SIDE_LEN  = 10;

    for (std::size_t offset = 0; offset < total; ) {
        std::size_t n = std::min(bs, total - offset);

        std::vector<SQL_TIMESTAMP_STRUCT> ts(n);
        std::vector<std::array<SQLCHAR, EXCH_LEN + 1>>  exch(n);
        std::vector<std::array<SQLCHAR, SYM_LEN + 1>>   sym(n);
        std::vector<std::array<SQLCHAR, MTYPE_LEN + 1>> mtype(n);
        std::vector<std::array<SQLCHAR, TID_LEN + 1>>   tid(n);
        std::vector<std::array<SQLCHAR, SIDE_LEN + 1>>  side(n);

        std::vector<double> price(n), qty(n);
        std::vector<SQLCHAR> buyer_maker(n);

        std::vector<SQLLEN> ind_ts(n), ind_exch(n), ind_sym(n),
                            ind_mtype(n), ind_tid(n),
                            ind_price(n), ind_qty(n),
                            ind_side(n), ind_bm(n);

        for (std::size_t i = 0; i < n; ++i) {
            const auto& t = trades[offset + i];

            // t.timestamp_ms holds microseconds per your recent changes
            ts[i] = ts_from_epoch_us(t.timestamp_ms);

            copyStrFixed(t.exchange,    exch[i]);
            copyStrFixed(t.symbol,      sym[i]);
            copyStrFixed(t.market_type, mtype[i]);
            copyStrFixed(t.trade_id,    tid[i]);
            copyStrFixed(t.side,        side[i]);

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

        SQLULEN paramset_size = n;
        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAMSET_SIZE,
                       &paramset_size, 0);

        // 1: timestamp
        ret = SQLBindParameter(
            stmt_, 1, SQL_PARAM_INPUT,
            SQL_C_TYPE_TIMESTAMP, SQL_TYPE_TIMESTAMP,
            27, 6,
            ts.data(), sizeof(SQL_TIMESTAMP_STRUCT),
            ind_ts.data());

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
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades mkt");

        // 5: trade_id
        ret = SQLBindParameter(
            stmt_, 5, SQL_PARAM_INPUT,
            SQL_C_CHAR, SQL_VARCHAR,
            TID_LEN, 0,
            tid[0].data(), TID_LEN + 1,
            ind_tid.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades tid");

        // 6: price
        ret = SQLBindParameter(
            stmt_, 6, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            38, 18, price.data(), 0, ind_price.data());
        if (!SQL_SUCCEEDED(ret))
            throwODBCError(SQL_HANDLE_STMT, stmt_, "Bind trades price");

        // 7: quantity
        ret = SQLBindParameter(
            stmt_, 7, SQL_PARAM_INPUT,
            SQL_C_DOUBLE, SQL_DECIMAL,
            38, 18, qty.data(), 0, ind_qty.data());
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

    if (candles.empty()) return;

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

    for (std::size_t offset = 0; offset < total; ) {
        std::size_t n = std::min(bs, total - offset);

        SQLSetStmtAttr(stmt_, SQL_ATTR_PARAMSET_SIZE,
                       (SQLPOINTER)n, 0);

        std::vector<SQL_TIMESTAMP_STRUCT> ts(n);
        std::vector<std::array<SQLCHAR, STR_LEN + 1>> exch(n), sym(n),
                                                     mtype(n), tf(n);
        std::vector<double> o(n), h(n), l(n), c(n), v(n);

        std::vector<SQLLEN> ind_ts(n), ind_exch(n), ind_sym(n),
                            ind_mtype(n), ind_tf(n),
                            ind_o(n), ind_h(n), ind_l(n),
                            ind_c(n), ind_v(n);

        for (std::size_t i = 0; i < n; ++i) {
            const auto& cd = candles[offset + i];

            ts[i] = ts_from_epoch_us(cd.timestamp_ms);
            ind_ts[i] = sizeof(SQL_TIMESTAMP_STRUCT);

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
            SQL_C_TYPE_TIMESTAMP, SQL_TYPE_TIMESTAMP,
            27, 6, ts.data(), sizeof(SQL_TIMESTAMP_STRUCT), ind_ts.data());
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
    const std::vector<MarketData::OrderbookSnapshot>& obs,
    std::size_t batch_size) {

    if (obs.empty()) return;
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

        std::vector<SQL_TIMESTAMP_STRUCT> ts(n);
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

            ts[i]     = makeSqlTsFromEpochUs(ob.timestamp_ms);
            ind_ts[i] = sizeof(SQL_TIMESTAMP_STRUCT);

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
            SQL_C_TYPE_TIMESTAMP, SQL_TYPE_TIMESTAMP,
            27, 6,
            ts.data(), sizeof(SQL_TIMESTAMP_STRUCT),
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
            throwODBCError(SQL_HANDLE_STMT, stmt_,
                           "SQLExecute bulkInsertOrderbooks");
        }

        SQLFreeStmt(stmt_, SQL_RESET_PARAMS);
        offset += n;
    }

    commitTransaction();
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
