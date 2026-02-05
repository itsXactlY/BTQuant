#include <QPainter>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QHeaderView>
#include <QDebug>

#include "components/pricestatisticpanel.h"

PriceStatisticPanel::PriceStatisticPanel(QWidget *parent)
    : QTableWidget(parent)
    , m_rowHeight(30)
    , m_visibleRows(0)
    , m_totalRows(0)
    , m_firstVisibleRow(0)
{
    initializeUI();
    connectScrollSignals();
}

void PriceStatisticPanel::initializeUI()
{
    // Set up the table properties
    setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    
    // Disable default vertical scrollbar as we'll implement custom virtual scrolling
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    
    // Enable alternating row colors for better readability
    setAlternatingRowColors(true);
    
    // Set selection behavior
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setSelectionMode(QAbstractItemView::SingleSelection);
    
    // Initialize headers
    setupHeaders();
    
    // Calculate visible rows based on widget height
    updateVisibleRows();
}

void PriceStatisticPanel::setupHeaders()
{
    QStringList headers;
    headers << "Symbol" << "Last Price" << "Change" << "Change %" << "Volume" << "High" << "Low" << "Open" << "Close";
    setColumnCount(headers.size());
    setHorizontalHeaderLabels(headers);
    
    // Resize columns to fit content
    horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
}

void PriceStatisticPanel::connectScrollSignals()
{
    // Create a custom scrollbar for virtual scrolling
    m_virtualScrollBar = new QScrollBar(Qt::Vertical, this);
    
    // Connect scroll bar signals to handle virtual scrolling
    connect(m_virtualScrollBar, &QScrollBar::valueChanged, this, &PriceStatisticPanel::onScrollValueChanged);
}

void PriceStatisticPanel::updateVisibleRows()
{
    int tableHeight = height();
    m_visibleRows = tableHeight / m_rowHeight + 2; // +2 for buffer
    
    // Update scrollbar range based on total rows
    if (m_totalRows > m_visibleRows) {
        m_virtualScrollBar->setRange(0, m_totalRows - m_visibleRows);
    } else {
        m_virtualScrollBar->setRange(0, 0);
    }
    
    // Update the scrollbar geometry
    updateScrollbarGeometry();
}

void PriceStatisticPanel::updateScrollbarGeometry()
{
    int scrollBarWidth = 16; // Standard scrollbar width
    int x = width() - scrollBarWidth;
    int y = horizontalHeader()->geometry().bottom();
    int h = height() - horizontalHeader()->height();
    
    m_virtualScrollBar->setGeometry(x, y, scrollBarWidth, h);
}

void PriceStatisticPanel::setData(const QVector<PriceStatData> &data)
{
    m_data = data;
    m_totalRows = data.size();
    
    // Update the virtual scrollbar range
    updateVisibleRows();
    
    // Refresh the visible portion of the table
    refreshVisibleData();
}

void PriceStatisticPanel::refreshVisibleData()
{
    // Clear current items
    clearContents();
    setRowCount(qMin(m_visibleRows, m_totalRows));
    
    // Populate visible rows based on m_firstVisibleRow
    int endRow = qMin(m_firstVisibleRow + m_visibleRows, m_totalRows);
    for (int i = m_firstVisibleRow, row = 0; i < endRow; ++i, ++row) {
        const PriceStatData &stat = m_data[i];
        
        setItem(row, 0, new QTableWidgetItem(stat.symbol));
        setItem(row, 1, new QTableWidgetItem(QString::number(stat.lastPrice, 'f', 2)));
        
        // Format change with color coding
        auto changeItem = new QTableWidgetItem(QString::number(stat.change, 'f', 2));
        changeItem->setForeground(stat.change >= 0 ? Qt::green : Qt::red);
        setItem(row, 2, changeItem);
        
        // Format change percentage with color coding
        auto changePercentItem = new QTableWidgetItem(QString::number(stat.changePercent, 'f', 2) + "%");
        changePercentItem->setForeground(stat.changePercent >= 0 ? Qt::green : Qt::red);
        setItem(row, 3, changePercentItem);
        
        setItem(row, 4, new QTableWidgetItem(QString::number(stat.volume)));
        setItem(row, 5, new QTableWidgetItem(QString::number(stat.high, 'f', 2)));
        setItem(row, 6, new QTableWidgetItem(QString::number(stat.low, 'f', 2)));
        setItem(row, 7, new QTableWidgetItem(QString::number(stat.open, 'f', 2)));
        setItem(row, 8, new QTableWidgetItem(QString::number(stat.close, 'f', 2)));
    }
}

void PriceStatisticPanel::onScrollValueChanged(int value)
{
    m_firstVisibleRow = value;
    refreshVisibleData();
}

void PriceStatisticPanel::resizeEvent(QResizeEvent *event)
{
    // Update visible rows count when widget is resized
    updateVisibleRows();
    
    // Update scrollbar geometry
    updateScrollbarGeometry();
    
    // Refresh visible data
    refreshVisibleData();
    
    QTableWidget::resizeEvent(event);
}

void PriceStatisticPanel::wheelEvent(QWheelEvent *event)
{
    // Handle mouse wheel events for virtual scrolling
    int delta = event->angleDelta().y();
    int scrollSteps = delta / 120; // Standard wheel step is 120
    
    // Adjust the scrollbar value based on wheel movement
    m_virtualScrollBar->setValue(m_virtualScrollBar->value() - scrollSteps);
    
    event->accept();
}

// Public methods for interacting with the panel
void PriceStatisticPanel::addData(const PriceStatData &data)
{
    m_data.append(data);
    m_totalRows = m_data.size();
    
    updateVisibleRows();
    refreshVisibleData();
}

void PriceStatisticPanel::clearAllData()
{
    m_data.clear();
    m_totalRows = 0;
    m_firstVisibleRow = 0;
    
    setRowCount(0);
    
    updateVisibleRows();
}

void PriceStatisticPanel::updateData(int index, const PriceStatData &data)
{
    if (index >= 0 && index < m_data.size()) {
        m_data[index] = data;
        
        // If the updated data is within the visible range, refresh it
        if (index >= m_firstVisibleRow && index < m_firstVisibleRow + m_visibleRows) {
            refreshVisibleData();
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