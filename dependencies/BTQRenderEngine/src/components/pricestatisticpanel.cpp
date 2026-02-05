#include <QPainter>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QHeaderView>
#include <QDebug>
#include <QApplication>
#include <QLinearGradient>
#include <QStyleOption>
#include <cmath>

#include "components/pricestatisticpanel.h"

PriceStatisticPanel::PriceStatisticPanel(QWidget *parent)
    : QWidget(parent)
    , m_rowHeight(30)
    , m_visibleRows(0)
    , m_totalRows(0)
    , m_firstVisibleRow(0)
    , m_selectedRow(-1)
    , m_sortColumn(-1)
    , m_sortOrder(Qt::AscendingOrder)
    , m_showGridLines(true)
    , m_alternateRowColors(true)
    , m_headerHeight(30)
{
    initializeUI();
    connectScrollSignals();
}

void PriceStatisticPanel::initializeUI()
{
    // Set up the widget properties
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    setMouseTracking(true); // Enable hover effects

    // Create and configure the virtual scrollbar
    m_virtualScrollBar = new QScrollBar(Qt::Vertical, this);
    m_virtualScrollBar->hide(); // Initially hidden until data is loaded

    // Initialize headers
    setupHeaders();

    // Calculate visible rows based on widget height
    updateVisibleRows();
}

void PriceStatisticPanel::setupHeaders()
{
    m_headers << "Symbol" << "Last Price" << "Change" << "Change %" << "Volume" << "High" << "Low" << "Open" << "Close";
    m_columnCount = m_headers.size();

    // Define column types for specialized formatting
    m_columnTypes.resize(m_columnCount);
    m_columnTypes[0] = TextColumn;       // Symbol
    m_columnTypes[1] = NumericColumn;    // Last Price
    m_columnTypes[2] = NumericColumn;    // Change
    m_columnTypes[3] = PercentageColumn; // Change %
    m_columnTypes[4] = NumericColumn;    // Volume
    m_columnTypes[5] = NumericColumn;    // High
    m_columnTypes[6] = NumericColumn;    // Low
    m_columnTypes[7] = NumericColumn;    // Open
    m_columnTypes[8] = NumericColumn;    // Close

    // Calculate column widths based on widget width
    updateColumnWidths();
}

void PriceStatisticPanel::updateColumnWidths()
{
    int totalWidth = width();
    int scrollbarWidth = m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0;
    int availableWidth = totalWidth - scrollbarWidth;

    m_columnWidths.clear();
    
    // Define proportional widths for different columns
    QVector<double> proportions = {0.12, 0.12, 0.10, 0.10, 0.12, 0.10, 0.10, 0.12, 0.12};
    
    for (int i = 0; i < m_columnCount; ++i) {
        m_columnWidths.append(static_cast<int>(availableWidth * proportions[i]));
    }
    
    // Adjust for rounding errors by adding remainder to the last column
    int totalAllocated = 0;
    for (int width : m_columnWidths) {
        totalAllocated += width;
    }
    if (totalAllocated != availableWidth) {
        m_columnWidths[m_columnWidths.size()-1] += availableWidth - totalAllocated;
    }
}

void PriceStatisticPanel::connectScrollSignals()
{
    // Connect scroll bar signals to handle virtual scrolling
    connect(m_virtualScrollBar, &QScrollBar::valueChanged, this, &PriceStatisticPanel::onScrollValueChanged);
}

void PriceStatisticPanel::updateVisibleRows()
{
    int widgetHeight = height();
    m_visibleRows = widgetHeight / m_rowHeight + 2; // +2 for buffer

    // Update scrollbar range based on total rows
    if (m_totalRows > m_visibleRows) {
        m_virtualScrollBar->setRange(0, m_totalRows - m_visibleRows);
        m_virtualScrollBar->show();
    } else {
        m_virtualScrollBar->setRange(0, 0);
        m_virtualScrollBar->hide();
    }

    // Update the scrollbar geometry
    updateScrollbarGeometry();
}

void PriceStatisticPanel::updateScrollbarGeometry()
{
    int scrollbarWidth = 16; // Standard scrollbar width
    int x = width() - scrollbarWidth;
    int y = m_headerHeight; // Start below header
    int h = height() - m_headerHeight;

    m_virtualScrollBar->setGeometry(x, y, scrollbarWidth, h);
}

void PriceStatisticPanel::setData(const QVector<PriceStatData> &data)
{
    m_data = data;
    m_totalRows = data.size();

    // Update the virtual scrollbar range
    updateVisibleRows();

    // Update column widths based on available space
    updateColumnWidths();

    // Repaint the widget
    update();
}

void PriceStatisticPanel::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Paint background
    painter.fillRect(rect(), palette().color(QPalette::Base));

    // Draw headers
    paintHeaders(painter);

    // Draw grid lines if enabled
    if (m_showGridLines) {
        paintGridLines(painter);
    }

    // Draw data rows
    paintDataRows(painter);

    // Draw selection highlight if needed
    if (m_selectedRow >= m_firstVisibleRow && m_selectedRow < m_firstVisibleRow + m_visibleRows) {
        paintSelectionHighlight(painter, m_selectedRow - m_firstVisibleRow);
    }

    QWidget::paintEvent(event);
}

void PriceStatisticPanel::paintHeaders(QPainter &painter)
{
    QRect headerRect(0, 0, width() - (m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0), m_headerHeight);

    // Paint header background with gradient
    QLinearGradient gradient(0, 0, 0, m_headerHeight);
    gradient.setColorAt(0, QColor(240, 240, 240));
    gradient.setColorAt(1, QColor(220, 220, 220));
    painter.fillRect(headerRect, gradient);

    // Draw header separator
    painter.setPen(QPen(QColor(200, 200, 200), 1));
    painter.drawLine(0, m_headerHeight - 1, headerRect.width(), m_headerHeight - 1);

    // Draw sort indicator if sorting is active
    if (m_sortColumn >= 0) {
        QRect sortIndicatorRect = getColumnRect(m_sortColumn, 0);
        sortIndicatorRect.adjust(0, 0, -5, 0);
        
        // Draw triangle for sort direction
        QPoint points[3];
        if (m_sortOrder == Qt::AscendingOrder) {
            // Upward triangle
            points[0] = QPoint(sortIndicatorRect.right() - 10, sortIndicatorRect.top() + 10);
            points[1] = QPoint(sortIndicatorRect.right() - 5, sortIndicatorRect.top() + 5);
            points[2] = QPoint(sortIndicatorRect.right(), sortIndicatorRect.top() + 10);
        } else {
            // Downward triangle
            points[0] = QPoint(sortIndicatorRect.right() - 10, sortIndicatorRect.top() + 5);
            points[1] = QPoint(sortIndicatorRect.right() - 5, sortIndicatorRect.top() + 10);
            points[2] = QPoint(sortIndicatorRect.right(), sortIndicatorRect.top() + 5);
        }
        
        painter.setBrush(QColor(100, 100, 100));
        painter.drawPolygon(points, 3);
    }

    // Draw header text
    painter.setPen(QPen(Qt::black, 1));
    QFont headerFont = painter.font();
    headerFont.setBold(true);
    painter.setFont(headerFont);

    for (int col = 0; col < m_columnCount; ++col) {
        QRect cellRect = getColumnRect(col, 0);
        cellRect.adjust(5, 0, -5, 0); // Add padding
        painter.drawText(cellRect, Qt::AlignLeft | Qt::AlignVCenter, m_headers[col]);
    }

    // Reset font
    headerFont.setBold(false);
    painter.setFont(headerFont);
}

void PriceStatisticPanel::paintGridLines(QPainter &painter)
{
    // Vertical grid lines
    int currentX = 0;
    for (int i = 0; i <= m_columnCount; ++i) {
        if (i < m_columnWidths.size()) {
            currentX += m_columnWidths[i];
        }
        
        painter.setPen(QPen(QColor(230, 230, 230), 1));
        painter.drawLine(currentX, m_headerHeight, currentX, height());
    }

    // Horizontal grid lines
    int startY = m_headerHeight;
    int endRow = qMin(m_firstVisibleRow + m_visibleRows, m_totalRows);
    
    for (int i = m_firstVisibleRow, row = 0; i < endRow; ++i, ++row) {
        int yPos = startY + row * m_rowHeight;
        painter.setPen(QPen(QColor(230, 230, 230), 1));
        painter.drawLine(0, yPos, width() - (m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0), yPos);
    }
}

void PriceStatisticPanel::paintDataRows(QPainter &painter)
{
    int startY = m_headerHeight; // Skip header row
    int endRow = qMin(m_firstVisibleRow + m_visibleRows, m_totalRows);

    for (int i = m_firstVisibleRow, row = 0; i < endRow; ++i, ++row) {
        int yPos = startY + row * m_rowHeight;

        // Paint alternate row background if enabled
        if (m_alternateRowColors && row % 2 == 0) {
            painter.fillRect(0, yPos, width() - (m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0), m_rowHeight,
                           QColor(248, 248, 248));
        }

        if (i < m_data.size()) {
            const PriceStatData &stat = m_data[i];

            // Draw cell contents
            paintCell(painter, stat, row, yPos);
        }
    }
}

void PriceStatisticPanel::paintCell(QPainter &painter, const PriceStatData &stat, int row, int yPos)
{
    // Column 0: Symbol
    QRect symbolRect = getColumnRect(0, row + 1);
    symbolRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(symbolRect, Qt::AlignLeft | Qt::AlignVCenter, stat.symbol);

    // Column 1: Last Price
    QRect lastPriceRect = getColumnRect(1, row + 1);
    lastPriceRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(lastPriceRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.lastPrice, 'f', 2));

    // Column 2: Change
    QRect changeRect = getColumnRect(2, row + 1);
    changeRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(stat.change >= 0 ? QPen(Qt::darkGreen, 1) : QPen(Qt::darkRed, 1));
    painter.drawText(changeRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.change, 'f', 2));

    // Column 3: Change %
    QRect changePercentRect = getColumnRect(3, row + 1);
    changePercentRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(stat.change >= 0 ? QPen(Qt::darkGreen, 1) : QPen(Qt::darkRed, 1));
    painter.drawText(changePercentRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.changePercent, 'f', 2) + "%");

    // Column 4: Volume
    QRect volumeRect = getColumnRect(4, row + 1);
    volumeRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(volumeRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.volume));

    // Column 5: High
    QRect highRect = getColumnRect(5, row + 1);
    highRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(highRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.high, 'f', 2));

    // Column 6: Low
    QRect lowRect = getColumnRect(6, row + 1);
    lowRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(lowRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.low, 'f', 2));

    // Column 7: Open
    QRect openRect = getColumnRect(7, row + 1);
    openRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(openRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.open, 'f', 2));

    // Column 8: Close
    QRect closeRect = getColumnRect(8, row + 1);
    closeRect.adjust(5, 0, -5, 0); // Add padding
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(closeRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.close, 'f', 2));
}

void PriceStatisticPanel::paintSelectionHighlight(QPainter &painter, int visualRow)
{
    int yPos = m_headerHeight + visualRow * m_rowHeight;
    int contentWidth = width() - (m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0);

    // Draw a subtle selection highlight
    QLinearGradient gradient(0, yPos, 0, yPos + m_rowHeight);
    gradient.setColorAt(0, QColor(180, 210, 255, 150));
    gradient.setColorAt(1, QColor(150, 190, 240, 150));
    
    painter.fillRect(0, yPos, contentWidth, m_rowHeight, gradient);
}

QRect PriceStatisticPanel::getColumnRect(int column, int row) const
{
    int xPos = 0;
    for (int i = 0; i < column; ++i) {
        xPos += m_columnWidths.value(i, 0);
    }

    int yPos = row * m_rowHeight;
    int contentWidth = width() - (m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0);

    // For the last column, use remaining width to fill any gaps
    int colWidth = (column == m_columnCount - 1) ?
                   contentWidth - xPos :
                   m_columnWidths.value(column, 0);

    return QRect(xPos, yPos, colWidth, m_rowHeight);
}

void PriceStatisticPanel::onScrollValueChanged(int value)
{
    m_firstVisibleRow = value;
    update(); // Trigger repaint instead of refreshing data
}

void PriceStatisticPanel::resizeEvent(QResizeEvent *event)
{
    // Update visible rows count when widget is resized
    updateVisibleRows();

    // Update column widths based on new available space
    updateColumnWidths();

    // Update scrollbar geometry
    updateScrollbarGeometry();

    QWidget::resizeEvent(event);
}

void PriceStatisticPanel::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        int yPos = event->pos().y();

        // Check if click is in header area
        if (yPos < m_headerHeight) {
            // Header click - implement sorting
            int clickedColumn = getColumnAtPosition(event->pos().x());
            if (clickedColumn >= 0) {
                // Toggle sort order if clicking the same column, otherwise set new sort column
                if (m_sortColumn == clickedColumn) {
                    m_sortOrder = (m_sortOrder == Qt::AscendingOrder) ? Qt::DescendingOrder : Qt::AscendingOrder;
                } else {
                    m_sortColumn = clickedColumn;
                    m_sortOrder = Qt::AscendingOrder;
                }
                
                // Sort the data
                sortData();
                
                // Emit signal for header click
                emit headerClicked(clickedColumn);
            }
        } else {
            // Data row click
            int rowOffset = (yPos - m_headerHeight) / m_rowHeight;
            int actualRow = m_firstVisibleRow + rowOffset;

            if (actualRow < m_totalRows) {
                m_selectedRow = actualRow;
                update(); // Trigger repaint to show selection

                // Emit signal for the selected row
                emit rowSelected(actualRow);
            }
        }
    }

    QWidget::mousePressEvent(event);
}

void PriceStatisticPanel::sortData()
{
    if (m_sortColumn < 0 || m_data.isEmpty()) return;

    std::sort(m_data.begin(), m_data.end(), [this](const PriceStatData &a, const PriceStatData &b) {
        bool ascending = (m_sortOrder == Qt::AscendingOrder);
        
        switch (m_sortColumn) {
            case 0: // Symbol
                return ascending ? (a.symbol < b.symbol) : (a.symbol > b.symbol);
            case 1: // Last Price
                return ascending ? (a.lastPrice < b.lastPrice) : (a.lastPrice > b.lastPrice);
            case 2: // Change
                return ascending ? (a.change < b.change) : (a.change > b.change);
            case 3: // Change %
                return ascending ? (a.changePercent < b.changePercent) : (a.changePercent > b.changePercent);
            case 4: // Volume
                return ascending ? (a.volume < b.volume) : (a.volume > b.volume);
            case 5: // High
                return ascending ? (a.high < b.high) : (a.high > b.high);
            case 6: // Low
                return ascending ? (a.low < b.low) : (a.low > b.low);
            case 7: // Open
                return ascending ? (a.open < b.open) : (a.open > b.open);
            case 8: // Close
                return ascending ? (a.close < b.close) : (a.close > b.close);
            default:
                return false;
        }
    });
    
    update();
}

int PriceStatisticPanel::getColumnAtPosition(int x) const
{
    int currentX = 0;
    for (int i = 0; i < m_columnWidths.size(); ++i) {
        if (x >= currentX && x < currentX + m_columnWidths[i]) {
            return i;
        }
        currentX += m_columnWidths[i];
    }
    return -1;
}

void PriceStatisticPanel::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
        case Qt::Key_Up:
            if (m_selectedRow > 0) {
                m_selectedRow--;
                ensureRowVisible(m_selectedRow);
                update();
                emit rowSelected(m_selectedRow);
            }
            break;
        case Qt::Key_Down:
            if (m_selectedRow < m_totalRows - 1) {
                m_selectedRow++;
                ensureRowVisible(m_selectedRow);
                update();
                emit rowSelected(m_selectedRow);
            }
            break;
        case Qt::Key_PageUp:
            m_selectedRow = qMax(0, m_selectedRow - m_visibleRows);
            ensureRowVisible(m_selectedRow);
            update();
            emit rowSelected(m_selectedRow);
            break;
        case Qt::Key_PageDown:
            m_selectedRow = qMin(m_totalRows - 1, m_selectedRow + m_visibleRows);
            ensureRowVisible(m_selectedRow);
            update();
            emit rowSelected(m_selectedRow);
            break;
        case Qt::Key_Home:
            m_selectedRow = 0;
            ensureRowVisible(m_selectedRow);
            update();
            emit rowSelected(m_selectedRow);
            break;
        case Qt::Key_End:
            m_selectedRow = m_totalRows - 1;
            ensureRowVisible(m_selectedRow);
            update();
            emit rowSelected(m_selectedRow);
            break;
        default:
            QWidget::keyPressEvent(event);
            break;
    }
}

void PriceStatisticPanel::ensureRowVisible(int row)
{
    if (row < m_firstVisibleRow) {
        m_firstVisibleRow = row;
        m_virtualScrollBar->setValue(row);
    } else if (row >= m_firstVisibleRow + m_visibleRows) {
        m_firstVisibleRow = row - m_visibleRows + 1;
        m_virtualScrollBar->setValue(m_firstVisibleRow);
    }
}

void PriceStatisticPanel::wheelEvent(QWheelEvent *event)
{
    // Handle mouse wheel events for virtual scrolling
    int delta = event->angleDelta().y();
    int scrollSteps = delta / 120; // Standard wheel step is 120

    // Adjust the scrollbar value based on wheel movement
    int newValue = m_virtualScrollBar->value() - scrollSteps;
    m_virtualScrollBar->setValue(qBound(0, newValue, m_virtualScrollBar->maximum()));

    event->accept();
}

// Public methods for interacting with the panel
void PriceStatisticPanel::addData(const PriceStatData &data)
{
    m_data.append(data);
    m_totalRows = m_data.size();

    // Update the virtual scrollbar range
    updateVisibleRows();

    // Update column widths
    updateColumnWidths();

    // Trigger repaint
    update();
}

void PriceStatisticPanel::clearAllData()
{
    m_data.clear();
    m_totalRows = 0;
    m_firstVisibleRow = 0;
    m_selectedRow = -1;
    m_sortColumn = -1;

    // Update the virtual scrollbar range
    updateVisibleRows();

    // Trigger repaint
    update();
}

void PriceStatisticPanel::updateData(int index, const PriceStatData &data)
{
    if (index >= 0 && index < m_data.size()) {
        m_data[index] = data;

        // Only repaint if the updated row is visible
        if (index >= m_firstVisibleRow && index < m_firstVisibleRow + m_visibleRows) {
            update();
        }
    }
}

QVector<PriceStatData> PriceStatisticPanel::getData() const
{
    return m_data;
}

int PriceStatisticPanel::getTotalRowCount() const
{
    return m_totalRows;
}

int PriceStatisticPanel::getSelectedRow() const
{
    return m_selectedRow;
}

// Analytical methods
void PriceStatisticPanel::enableGridLines(bool enable)
{
    m_showGridLines = enable;
    update();
}

void PriceStatisticPanel::enableAlternateRowColors(bool enable)
{
    m_alternateRowColors = enable;
    update();
}

void PriceStatisticPanel::setSortIndicator(int column, Qt::SortOrder order)
{
    m_sortColumn = column;
    m_sortOrder = order;
    update();
}

Qt::SortOrder PriceStatisticPanel::getSortOrder() const
{
    return m_sortOrder;
}

int PriceStatisticPanel::getSortColumn() const
{
    return m_sortColumn;
}

void PriceStatisticPanel::applyFilter(const QString &filterText)
{
    // This would implement filtering logic
    // For now, we'll just trigger a refresh
    update();
}

void PriceStatisticPanel::calculateStatistics(double &avgPrice, double &volatility, double &priceRange) const
{
    if (m_data.isEmpty()) {
        avgPrice = 0.0;
        volatility = 0.0;
        priceRange = 0.0;
        return;
    }
    
    double sum = 0.0;
    double minPrice = m_data[0].lastPrice;
    double maxPrice = m_data[0].lastPrice;
    
    for (const auto &item : m_data) {
        sum += item.lastPrice;
        minPrice = qMin(minPrice, item.lastPrice);
        maxPrice = qMax(maxPrice, item.lastPrice);
    }
    
    avgPrice = sum / m_data.size();
    priceRange = maxPrice - minPrice;
    
    // Simple volatility calculation (standard deviation)
    double variance = 0.0;
    for (const auto &item : m_data) {
        double diff = item.lastPrice - avgPrice;
        variance += diff * diff;
    }
    variance /= m_data.size();
    volatility = sqrt(variance);
}