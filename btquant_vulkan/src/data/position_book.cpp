#include "position_book.hpp"

#include "trade_journal.hpp"
#include <cmath>

namespace btquant {

double PositionBook::averageEntry(double oldSize, double oldAvg,
                                  double fillQty,  double fillPrice) {
    if (oldSize <= 0.0) return fillPrice;
    if (fillQty <= 0.0) return oldAvg;
    double totalCost = oldSize * oldAvg + fillQty * fillPrice;
    double totalSize = oldSize + fillQty;
    if (totalSize <= 0.0) return 0.0;
    return totalCost / totalSize;
}

PositionBook::CloseResult PositionBook::applyClose(double openSize,
                                                   double openAvg,
                                                   double closeQty,
                                                   double closePrice,
                                                   bool   openIsLong) {
    CloseResult r{0.0, openSize, openAvg};
    if (openSize <= 0.0 || closeQty <= 0.0 || closePrice <= 0.0) return r;
    double qty       = std::min(closeQty, openSize);
    double priceDiff = openIsLong ? (closePrice - openAvg)
                                  : (openAvg - closePrice);
    r.realizedDelta  = qty * priceDiff;
    r.newSize        = openSize - qty;
    r.newAvg         = (r.newSize > 0.0) ? openAvg : 0.0;
    return r;
}

double PositionBook::fill(const std::string& symbol, bool isLong,
                          double qty, double price) {
    if (qty <= 0.0 || price <= 0.0) return 0.0;

    double realizedDelta = 0.0;

    // Symbol switch with open position: flatten the old one at the new
    // fill price (treating the fill as a closing print for the old symbol).
    if (!m_pos.symbol.empty() && m_pos.symbol != symbol &&
        m_pos.size > 0.0) {
        double oldRealized = flatten(price);
        realizedDelta += oldRealized;
    }

    m_pos.symbol = symbol;
    m_pos.fillCount++;

    if (m_pos.size <= 0.0) {
        // Opening from flat.
        m_pos.isLong   = isLong;
        m_pos.size     = qty;
        m_pos.avgEntry = price;
    } else if (m_pos.isLong == isLong) {
        // Averaging in same direction.
        m_pos.avgEntry = averageEntry(m_pos.size, m_pos.avgEntry, qty, price);
        m_pos.size    += qty;
    } else {
        // Opposite direction — close/reduce. A close against a long is a
        // SELL (isLong=false, openIsLong=true).
        double appliedClose = std::min(qty, m_pos.size);
        auto cr = applyClose(m_pos.size, m_pos.avgEntry, qty, price,
                             m_pos.isLong);
        realizedDelta      += cr.realizedDelta;
        m_pos.realizedPnL  += cr.realizedDelta;
        m_pos.size          = cr.newSize;
        m_pos.avgEntry      = cr.newAvg;
        // If the fill exceeded the open size, the residual flips the
        // position into the opposite side at the fill price.
        double leftover = qty - appliedClose;
        if (m_pos.size <= 0.0 && leftover > 0.0) {
            m_pos.isLong   = isLong;
            m_pos.size     = leftover;
            m_pos.avgEntry = price;
        }
    }

    if (std::fabs(realizedDelta) > 0.0) {
        m_realizedClosed += realizedDelta;
    }
    return realizedDelta;
}

double PositionBook::markToMarket(double currentPrice) {
    if (m_pos.size <= 0.0 || currentPrice <= 0.0) {
        m_pos.unrealizedPnL = 0.0;
        return 0.0;
    }
    double diff = m_pos.isLong ? (currentPrice - m_pos.avgEntry)
                               : (m_pos.avgEntry - currentPrice);
    m_pos.unrealizedPnL = m_pos.size * diff;
    return m_pos.unrealizedPnL;
}

double PositionBook::flatten(double price) {
    if (m_pos.size <= 0.0 || price <= 0.0) return 0.0;
    double realizedDelta = m_pos.size *
                           (m_pos.isLong ? (price - m_pos.avgEntry)
                                         : (m_pos.avgEntry - price));
    m_pos.realizedPnL += realizedDelta;
    m_realizedClosed  += realizedDelta;
    m_pos.size         = 0.0;
    m_pos.avgEntry     = 0.0;
    m_pos.unrealizedPnL = 0.0;
    return realizedDelta;
}

// Template definition kept here (not in the header) because the
// template parameter would otherwise force every translation unit that
// uses PositionBook to pull in TradeJournal's full definition.
template <typename JournalT>
size_t PositionBook::replay(const JournalT& journal, double* lastPrice) {
    m_pos = Position{};
    auto fills = journal.loadAll(nullptr);
    for (const auto& f : fills) {
        if (f.qty <= 0.0 || f.price <= 0.0) continue;
        fill(f.symbol, f.isLong, f.qty, f.price);
        if (lastPrice) *lastPrice = f.price;
    }
    return fills.size();
}

// Explicit instantiations for the journals we expect callers to use.
// Add new instantiations here if a different journal type is needed.
template size_t PositionBook::replay<TradeJournal>(
    const TradeJournal& journal, double* lastPrice);

} // namespace btquant
