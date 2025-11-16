#pragma once

#include <sql.h>
#include <sqlext.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_set>
#include <mutex>
#include "market_data_types.h"

class MSSQLBulkInserter {
public:
    explicit MSSQLBulkInserter(const std::string& connection_string);
    ~MSSQLBulkInserter();

    MSSQLBulkInserter(const MSSQLBulkInserter&) = delete;
    MSSQLBulkInserter& operator=(const MSSQLBulkInserter&) = delete;

    bool isConnected() const;

    void setAutoCommit(bool enabled);
    void beginTransaction();
    void commitTransaction();
    void rollbackTransaction();

    void bulkInsertTrades(const std::vector<MarketData::Trade>& trades,
                          std::size_t batch_size = 500);

    void bulkInsertOHLCV(const std::string& table_name,
                         const std::vector<MarketData::OHLCV>& candles,
                         std::size_t batch_size = 500);

    void bulkInsertOrderbooks(
        const std::vector<MarketData::OrderbookSnapshot>& snapshots,
        std::size_t batch_size = 200);

private:
    SQLHENV env_{SQL_NULL_HENV};
    SQLHDBC dbc_{SQL_NULL_HDBC};
    SQLHSTMT stmt_{SQL_NULL_HSTMT};
    std::string connection_string_;
    std::mutex ddl_mutex_;
    std::unordered_set<std::string> known_klines_tables_;

    void ensureCoreTables();                      // optional, for trades/orderbooks
    void ensureKlinesTable(const std::string&);   // per-symbol klines
    static std::string sanitizeIdentifier(const std::string& name);

    void resetStatement();
    void throwODBCError(SQLSMALLINT handle_type,
                        SQLHANDLE handle,
                        const std::string& context) const;

    static SQL_TIMESTAMP_STRUCT toSqlTimestamp(int64_t timestamp_us);
};
