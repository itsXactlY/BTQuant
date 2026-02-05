#include <QPainter>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QHeaderView>
#include <QDebug>
#include <QApplication>

#include "components/pricestatisticpanel.h"

PriceStatisticPanel::PriceStatisticPanel(QWidget *parent)
    : QWidget(parent)
    , m_rowHeight(30)
    , m_visibleRows(0)
    , m_totalRows(0)
    , m_firstVisibleRow(0)
    , m_selectedRow(-1)
{
    initializeUI();
    connectScrollSignals();
}

void PriceStatisticPanel::initializeUI()
{
    // Set up the widget properties
    setFocusPolicy(Qt::StrongFocus);
    setAttribute(Qt::WA_OpaquePaintEvent, true);
    
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
    
    // Calculate column widths based on widget width
    updateColumnWidths();
}

void PriceStatisticPanel::updateColumnWidths()
{
    int totalWidth = width();
    int scrollbarWidth = m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0;
    int availableWidth = totalWidth - scrollbarWidth;
    
    m_columnWidths.clear();
    for (int i = 0; i < m_columnCount; ++i) {
        // Distribute width evenly among columns
        m_columnWidths.append(availableWidth / m_columnCount);
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
    m_visibleRows = widgetHeight / m_rowHeight + 3; // +3 for buffer

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
    int y = 0; // Start from top since we're not using traditional headers
    int h = height();

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
    QRect headerRect(0, 0, width() - (m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0), m_rowHeight);
    
    // Paint header background
    painter.fillRect(headerRect, QColor(240, 240, 240));
    
    // Draw header separator
    painter.setPen(QPen(QColor(200, 200, 200), 1));
    painter.drawLine(0, m_rowHeight - 1, headerRect.width(), m_rowHeight - 1);
    
    // Draw header text
    painter.setPen(QPen(Qt::black, 1));
    QFont headerFont = painter.font();
    headerFont.setBold(true);
    painter.setFont(headerFont);
    
    for (int col = 0; col < m_columnCount; ++col) {
        QRect cellRect = getColumnRect(col, 0);
        painter.drawText(cellRect, Qt::AlignCenter, m_headers[col]);
    }
    
    // Reset font
    headerFont.setBold(false);
    painter.setFont(headerFont);
}

void PriceStatisticPanel::paintDataRows(QPainter &painter)
{
    int startY = m_rowHeight; // Skip header row
    int endRow = qMin(m_firstVisibleRow + m_visibleRows, m_totalRows);
    
    for (int i = m_firstVisibleRow, row = 0; i < endRow; ++i, ++row) {
        int yPos = startY + row * m_rowHeight;
        
        // Alternate row colors for better readability
        if (row % 2 == 0) {
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
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(symbolRect, Qt::AlignLeft | Qt::AlignVCenter, stat.symbol);

    // Column 1: Last Price
    QRect lastPriceRect = getColumnRect(1, row + 1);
    painter.drawText(lastPriceRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.lastPrice, 'f', 2));

    // Column 2: Change
    QRect changeRect = getColumnRect(2, row + 1);
    painter.setPen(stat.change >= 0 ? QPen(Qt::green, 1) : QPen(Qt::red, 1));
    painter.drawText(changeRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.change, 'f', 2));

    // Column 3: Change %
    QRect changePercentRect = getColumnRect(3, row + 1);
    painter.drawText(changePercentRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.changePercent, 'f', 2) + "%");

    // Column 4: Volume
    QRect volumeRect = getColumnRect(4, row + 1);
    painter.setPen(QPen(Qt::black, 1));
    painter.drawText(volumeRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.volume));

    // Column 5: High
    QRect highRect = getColumnRect(5, row + 1);
    painter.drawText(highRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.high, 'f', 2));

    // Column 6: Low
    QRect lowRect = getColumnRect(6, row + 1);
    painter.drawText(lowRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.low, 'f', 2));

    // Column 7: Open
    QRect openRect = getColumnRect(7, row + 1);
    painter.drawText(openRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.open, 'f', 2));

    // Column 8: Close
    QRect closeRect = getColumnRect(8, row + 1);
    painter.drawText(closeRect, Qt::AlignRight | Qt::AlignVCenter, QString::number(stat.close, 'f', 2));
}

void PriceStatisticPanel::paintSelectionHighlight(QPainter &painter, int visualRow)
{
    int yPos = m_rowHeight + visualRow * m_rowHeight;
    int contentWidth = width() - (m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0);
    
    painter.fillRect(0, yPos, contentWidth, m_rowHeight, QColor(180, 210, 255, 100));
}

QRect PriceStatisticPanel::getColumnRect(int column, int row) const
{
    int xPos = 0;
    for (int i = 0; i < column; ++i) {
        xPos += m_columnWidths[i];
    }
    
    int yPos = row * m_rowHeight;
    int contentWidth = width() - (m_virtualScrollBar->isVisible() ? m_virtualScrollBar->width() : 0);
    
    // For the last column, use remaining width to fill any gaps
    int colWidth = (column == m_columnCount - 1) ? 
                   contentWidth - xPos : 
                   m_columnWidths[column];
    
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
        if (yPos < m_rowHeight) {
            // Header click - could implement sorting here
            int clickedColumn = getColumnAtPosition(event->pos().x());
            emit headerClicked(clickedColumn);
        } else {
            // Data row click
            int rowOffset = (yPos - m_rowHeight) / m_rowHeight;
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
    m_virtualScrollBar->setValue(m_virtualScrollBar->value() - scrollSteps);

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