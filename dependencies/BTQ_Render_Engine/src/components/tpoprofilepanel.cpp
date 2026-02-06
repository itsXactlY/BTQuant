#include <QPainter>
#include <QPaintEvent>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QDebug>
#include <QTimer>
#include <QFontMetrics>
#include <QPen>
#include <QBrush>
#include <QRect>
#include <QPoint>
#include <QSize>
#include <QColor>
#include <QPalette>
#include <QStyle>
#include <QStyleOption>
#include <QMouseEvent>
#include <QContextMenuEvent>
#include <QMenu>
#include <QAction>
#include <QPolygon>

// Forward declarations or includes for TPO engine
#include "analytics/tpoengine.h"
#include "components/tpoprofilepanel.h"

// Include QObject header for Q_OBJECT macro
#include <QObject>

// Constructor implementation
TPOProfilePanel::TPOProfilePanel(QWidget *parent)
    : QWidget(parent)
    , m_minPrice(0.0)
    , m_maxPrice(100.0)
    , m_priceStep(0.25)
    , m_priceScaleWidth(60)
    , m_topMargin(20)
    , m_bottomMargin(20)
    , m_leftMargin(80)  // Extra space for price labels
    , m_rightMargin(20)
    , m_blockWidth(30)
    , m_blockHeight(20)
    , m_needsUpdate(true)
    , m_cached_poc(0.0)
    , m_cached_va_low(0.0)
    , m_cached_va_high(0.0)
    , m_cached_single_prints()
    , m_values_cached(false)
    , m_selectedPriceIndex(-1)
    , m_selectedTimeIndex(-1)
    , m_hasSelection(false)
{
    setMinimumSize(400, 300);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // Initialize default letter colors
    m_letterColors['A'] = QColor(255, 100, 100);  // Red
    m_letterColors['B'] = QColor(100, 255, 100);  // Green
    m_letterColors['C'] = QColor(100, 100, 255);  // Blue
    m_letterColors['D'] = QColor(255, 255, 100);  // Yellow
    m_letterColors['E'] = QColor(255, 100, 255);  // Magenta
    m_letterColors['F'] = QColor(100, 255, 255);  // Cyan
    m_letterColors['G'] = QColor(200, 150, 100);  // Brown
    m_letterColors['H'] = QColor(200, 100, 150);  // Pink
    m_letterColors['I'] = QColor(150, 200, 100);  // Light green
    m_letterColors['J'] = QColor(100, 150, 200);  // Light blue
    m_letterColors['K'] = QColor(255, 200, 150);  // Peach
    m_letterColors['L'] = QColor(200, 150, 255);  // Lavender
    m_letterColors['M'] = QColor(150, 255, 200);  // Mint
    m_letterColors['N'] = QColor(255, 150, 200);  // Salmon
    m_letterColors['O'] = QColor(200, 255, 150);  // Light yellow-green
    m_letterColors['P'] = QColor(150, 200, 255);  // Light blue
    m_letterColors['Q'] = QColor(220, 220, 220);  // Light gray
    m_letterColors['R'] = QColor(180, 180, 180);  // Medium gray
    m_letterColors['S'] = QColor(140, 140, 140);  // Dark gray
    m_letterColors['T'] = QColor(100, 100, 100);  // Very dark gray
    m_letterColors['U'] = QColor(255, 150, 100);  // Orange
    m_letterColors['V'] = QColor(150, 100, 255);  // Purple
    m_letterColors['W'] = QColor(100, 255, 150);  // Bright green
    m_letterColors['X'] = QColor(255, 100, 150);  // Rose
    m_letterColors['Y'] = QColor(150, 255, 100);  // Lime
    m_letterColors['Z'] = QColor(100, 150, 100);  // Olive

    // Initialize context menu
    m_contextMenu = new QMenu(this);
    m_splitProfileAction = new QAction("Split Profile", this);
    connect(m_splitProfileAction, &QAction::triggered, this, &TPOProfilePanel::splitProfileAction);
    m_contextMenu->addAction(m_splitProfileAction);

    // Connect timer for periodic updates if needed
    connect(new QTimer(this), &QTimer::timeout, this, &TPOProfilePanel::updateDisplay);
}

// Destructor implementation
TPOProfilePanel::~TPOProfilePanel()
{
    delete m_contextMenu;
}

void TPOProfilePanel::setData(const QVector<QMap<QString, QVariant>> &tpoData)
{
    m_tpoData = tpoData;

    // Process the TPO data to populate the TPO engine
    m_tpoEngine.clear();

    for (const auto &dataPoint : m_tpoData) {
        if (dataPoint.contains("price") && dataPoint.contains("time_index")) {
            double price = dataPoint["price"].toDouble();
            int timeIndex = dataPoint["time_index"].toInt();

            // Create a mock timestamp based on time index (in a real implementation, you'd have actual timestamps)
            auto timestamp = std::chrono::system_clock::now() + std::chrono::minutes(timeIndex * 30);

            // Create a PriceTick and process it
            PriceTick tick;
            tick.timestamp = timestamp;
            tick.price = price;
            tick.volume = 1.0; // Default volume for TPO counting

            m_tpoEngine.process_tick(tick);
        }
    }

    // Calculate POC, Value Area, and Single Prints
    const TPOProfile& profile = m_tpoEngine.get_tpo_profile();
    m_cached_poc = profile.get_poc();
    auto va = profile.get_value_area(70.0); // 70% of TPOs
    m_cached_va_low = va.first;
    m_cached_va_high = va.second;
    m_cached_single_prints = profile.get_single_print_levels(); // Cache single print levels
    m_values_cached = true;

    m_needsUpdate = true;
    update();
}

void TPOProfilePanel::setPriceRange(double minPrice, double maxPrice, double step)
{
    m_minPrice = minPrice;
    m_maxPrice = maxPrice;
    m_priceStep = step;
    m_needsUpdate = true;
    update();
}

void TPOProfilePanel::setTimePeriods(const QStringList &timeLabels)
{
    m_timeLabels = timeLabels;
    m_needsUpdate = true;
    update();
}

void TPOProfilePanel::setLetterColors(const QMap<QChar, QColor> &colors)
{
    m_letterColors = colors;
    m_needsUpdate = true;
    update();
}

void TPOProfilePanel::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw background
    painter.fillRect(rect(), palette().color(QPalette::Base));

    // Draw grid lines
    drawGridLines(painter);

    // Draw price scale on the left
    drawPriceScale(painter);

    // Draw TPO letter blocks
    drawLetterBlocks(painter);

    // Reset update flag
    m_needsUpdate = false;
}

void TPOProfilePanel::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    m_needsUpdate = true;
}

void TPOProfilePanel::updateDisplay()
{
    update();
}

void TPOProfilePanel::drawPriceScale(QPainter &painter)
{
    // Calculate available height for price scale
    int totalHeight = height() - m_topMargin - m_bottomMargin;
    int priceLevels = static_cast<int>((m_maxPrice - m_minPrice) / m_priceStep) + 1;

    if (priceLevels <= 0) return;

    m_blockHeight = totalHeight / priceLevels;

    // Draw the price scale background
    QRect scaleRect(m_leftMargin - m_priceScaleWidth, m_topMargin,
                    m_priceScaleWidth, totalHeight);
    painter.fillRect(scaleRect, QColor(245, 245, 245));
    painter.drawRect(scaleRect);

    // Draw price labels
    QFont font = painter.font();
    font.setPointSize(8);
    painter.setFont(font);

    // Draw price labels from top (highest price) to bottom (lowest price)
    for (int i = 0; i < priceLevels; ++i) {
        double price = m_maxPrice - (i * m_priceStep);

        // Calculate position for this price level
        int y = m_topMargin + (i * m_blockHeight);

        // Draw price label
        QString priceStr = QString::number(price, 'f', 2);
        QRect labelRect(m_leftMargin - m_priceScaleWidth + 5, y,
                        m_priceScaleWidth - 10, m_blockHeight);

        painter.drawText(labelRect, Qt::AlignRight | Qt::AlignVCenter, priceStr);

        // Draw horizontal grid line
        painter.setPen(QPen(QColor(220, 220, 220), 1));
        painter.drawLine(m_leftMargin, y, width() - m_rightMargin, y);
    }

    // Draw bottom line
    painter.setPen(QPen(QColor(220, 220, 220), 1));
    painter.drawLine(m_leftMargin, m_topMargin + totalHeight,
                     width() - m_rightMargin, m_topMargin + totalHeight);
}

void TPOProfilePanel::drawLetterBlocks(QPainter &painter)
{
    if (m_tpoData.isEmpty()) return;

    int totalHeight = height() - m_topMargin - m_bottomMargin;
    int priceLevels = static_cast<int>((m_maxPrice - m_minPrice) / m_priceStep) + 1;

    if (priceLevels <= 0 || m_timeLabels.isEmpty()) return;

    // Calculate block dimensions based on available space
    int availableWidth = width() - m_leftMargin - m_rightMargin;
    m_blockWidth = availableWidth / m_timeLabels.size();

    if (m_blockWidth < 10) m_blockWidth = 10;  // Minimum width

    // Iterate through TPO data and draw letter blocks
    for (int i = 0; i < m_tpoData.size(); ++i) {
        const auto &dataPoint = m_tpoData[i];

        if (!dataPoint.contains("price") || !dataPoint.contains("letter") ||
            !dataPoint.contains("time_index")) {
            continue;
        }

        double price = dataPoint["price"].toDouble();
        QChar letter = dataPoint["letter"].toString()[0];
        int timeIndex = dataPoint["time_index"].toInt();

        // Validate time index
        if (timeIndex < 0 || timeIndex >= m_timeLabels.size()) {
            continue;
        }

        // Get the price index
        int priceIndex = getPriceIndex(price);
        if (priceIndex < 0 || priceIndex >= priceLevels) continue;

        // Calculate block position
        int x = m_leftMargin + (timeIndex * m_blockWidth);
        int y = m_topMargin + (priceIndex * m_blockHeight);

        // Get color for this letter
        QColor color = m_letterColors.value(letter, QColor(200, 200, 200));

        // Draw the block
        QRect blockRect(x, y, m_blockWidth - 2, m_blockHeight - 2);
        painter.fillRect(blockRect, color);

        // Draw border
        painter.setPen(QPen(color.darker(), 1));
        painter.drawRect(blockRect);

        // Draw the letter
        painter.setPen(QPen(Qt::black, 1));
        QFont font = painter.font();
        font.setPointSize(8);
        painter.setFont(font);

        painter.drawText(blockRect, Qt::AlignCenter, QString(letter));
        
        // Check if this price level is a single print level and draw a marker
        bool isSinglePrint = false;
        double tolerance = m_priceStep / 2.0; // Half the price step for accurate matching
        for (double singlePrintPrice : m_cached_single_prints) {
            if (std::abs(singlePrintPrice - price) <= tolerance) { // Within the price bucket tolerance
                isSinglePrint = true;
                break;
            }
        }
        
        if (isSinglePrint) {
            // Draw a distinct marker for single print levels (e.g., a small triangle at the top)
            QPolygon triangle;
            int centerX = x + (m_blockWidth - 2) / 2;
            int centerY = y + 3; // Near the top of the block
            triangle << QPoint(centerX, centerY) 
                     << QPoint(centerX - 4, centerY + 6) 
                     << QPoint(centerX + 4, centerY + 6);
            
            painter.setPen(QPen(QColor(255, 0, 0), 2)); // Red color for single print marker
            painter.setBrush(QBrush(QColor(255, 0, 0))); // Fill the triangle
            painter.drawPolygon(triangle);
        }
    }

    // Highlight selected TPO bar if there's a selection
    if (m_hasSelection && m_selectedTimeIndex >= 0 && m_selectedTimeIndex < m_timeLabels.size() &&
        m_selectedPriceIndex >= 0 && m_selectedPriceIndex < priceLevels) {

        int x = m_leftMargin + (m_selectedTimeIndex * m_blockWidth);
        int y = m_topMargin + (m_selectedPriceIndex * m_blockHeight);
        QRect selectionRect(x, y, m_blockWidth - 2, m_blockHeight - 2);

        // Draw a highlight around the selected block
        QPen highlightPen(QColor(255, 0, 0), 3); // Thick red border
        painter.setPen(highlightPen);
        painter.drawRect(selectionRect);
    }

    // Draw Value Area if calculated
    if (m_values_cached && m_cached_va_low > 0 && m_cached_va_high > 0) {
        // Calculate Y positions for Value Area boundaries
        int va_low_y = m_topMargin + (getPriceIndex(m_cached_va_low) * m_blockHeight);
        int va_high_y = m_topMargin + (getPriceIndex(m_cached_va_high) * m_blockHeight);

        // Draw shaded area for Value Area
        QRect va_rect(m_leftMargin, va_high_y,
                      width() - m_leftMargin - m_rightMargin,
                      va_low_y - va_high_y);

        QColor va_color(255, 215, 0, 50); // Semi-transparent gold
        painter.fillRect(va_rect, va_color);

        // Draw Value Area boundaries
        QPen va_pen(QColor(255, 165, 0), 1); // Orange line
        painter.setPen(va_pen);
        painter.drawLine(m_leftMargin, va_low_y, width() - m_rightMargin, va_low_y);  // VAL
        painter.drawLine(m_leftMargin, va_high_y, width() - m_rightMargin, va_high_y); // VAH

        // Draw POC line if available
        if (m_cached_poc > 0) {
            int poc_y = m_topMargin + (getPriceIndex(m_cached_poc) * m_blockHeight);
            QPen poc_pen(QColor(255, 255, 0), 1); // Yellow line, 1px as requested
            painter.setPen(poc_pen);
            painter.drawLine(m_leftMargin, poc_y, width() - m_rightMargin, poc_y);  // POC
        }
    }
}

void TPOProfilePanel::drawGridLines(QPainter &painter)
{
    // Draw outer border
    painter.setPen(QPen(QColor(180, 180, 180), 2));
    QRect borderRect(m_leftMargin, m_topMargin,
                     width() - m_leftMargin - m_rightMargin,
                     height() - m_topMargin - m_bottomMargin);
    painter.drawRect(borderRect);

    // Draw vertical grid lines for time periods if we have them
    if (!m_timeLabels.isEmpty()) {
        int availableWidth = width() - m_leftMargin - m_rightMargin;
        int segmentWidth = availableWidth / m_timeLabels.size();

        painter.setPen(QPen(QColor(220, 220, 220), 1));

        for (int i = 1; i < m_timeLabels.size(); ++i) {
            int x = m_leftMargin + (i * segmentWidth);
            painter.drawLine(x, m_topMargin, x, height() - m_bottomMargin);
        }
    }
}

QRect TPOProfilePanel::getBlockRect(int priceIndex, int timeIndex) const
{
    int x = m_leftMargin + (timeIndex * m_blockWidth);
    int y = m_topMargin + (priceIndex * m_blockHeight);
    return QRect(x, y, m_blockWidth - 2, m_blockHeight - 2);
}

int TPOProfilePanel::getPriceIndex(double price) const
{
    if (price < m_minPrice || price > m_maxPrice) return -1;

    int index = static_cast<int>((m_maxPrice - price) / m_priceStep);
    int maxIndex = static_cast<int>((m_maxPrice - m_minPrice) / m_priceStep);

    return qBound(0, index, maxIndex);
}

double TPOProfilePanel::getPriceAt(int index) const
{
    return m_maxPrice - (index * m_priceStep);
}

void TPOProfilePanel::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {
        // Find which TPO block was clicked
        QPoint pos = event->pos();
        
        // Calculate available dimensions
        int totalHeight = height() - m_topMargin - m_bottomMargin;
        int priceLevels = static_cast<int>((m_maxPrice - m_minPrice) / m_priceStep) + 1;
        int availableWidth = width() - m_leftMargin - m_rightMargin;
        int timeSlots = m_timeLabels.size();
        
        if (priceLevels <= 0 || timeSlots <= 0) return;
        
        // Calculate block dimensions based on available space
        m_blockHeight = totalHeight / priceLevels;
        m_blockWidth = availableWidth / timeSlots;

        // Check if click is within the TPO display area
        if (pos.x() >= m_leftMargin && pos.x() < (m_leftMargin + availableWidth) &&
            pos.y() >= m_topMargin && pos.y() < (m_topMargin + totalHeight)) {
            
            // Calculate which price index and time index were clicked
            int clickedTimeIndex = (pos.x() - m_leftMargin) / m_blockWidth;
            int clickedPriceIndex = (pos.y() - m_topMargin) / m_blockHeight;
            
            // Validate indices
            if (clickedTimeIndex >= 0 && clickedTimeIndex < timeSlots &&
                clickedPriceIndex >= 0 && clickedPriceIndex < priceLevels) {
                
                // Store selection
                m_selectedPriceIndex = clickedPriceIndex;
                m_selectedTimeIndex = clickedTimeIndex;
                m_hasSelection = true;
                
                // Show context menu
                m_contextMenu->exec(event->globalPos());
            }
        }
    } else {
        // Handle other mouse buttons if needed
        QWidget::mousePressEvent(event);
    }
}

void TPOProfilePanel::contextMenuEvent(QContextMenuEvent *event)
{
    // This is handled in mousePressEvent to have more control over when the menu appears
    // Just call the parent implementation to prevent default context menu
    QWidget::contextMenuEvent(event);
}

void TPOProfilePanel::splitProfileAction()
{
    // This function will split the selected TPO bar into individual sub-period bars
    if (!m_hasSelection) return;

    // Get the selected time period
    int timeIndex = m_selectedTimeIndex;

    // Get the corresponding time label for the selected time index
    if (timeIndex < 0 || timeIndex >= m_timeLabels.size()) {
        m_hasSelection = false;
        m_selectedPriceIndex = -1;
        m_selectedTimeIndex = -1;
        return;
    }

    // Get the original time label
    QString originalTimeLabel = m_timeLabels[timeIndex];

    // Create new sub-period labels (for example, split one 30-min period into two 15-min periods)
    QString subPeriod1Label = originalTimeLabel + "_A";
    QString subPeriod2Label = originalTimeLabel + "_B";

    // Create new TPO data by duplicating the data for the selected time period
    // and assigning it to the new sub-periods
    QVector<QMap<QString, QVariant>> newTpoData;

    // First, copy all data points, adjusting time indices for those that come after the split point
    for (const auto &dataPoint : m_tpoData) {
        int current_time_index = dataPoint["time_index"].toInt();
        QMap<QString, QVariant> newDataPoint = dataPoint;

        if (current_time_index == timeIndex) {
            // This data point belongs to the time period being split
            // Create two copies for the new sub-periods
            QMap<QString, QVariant> subPeriod1Data = dataPoint;
            subPeriod1Data["time_index"] = timeIndex; // First sub-period takes the original index
            subPeriod1Data["subperiod_label"] = subPeriod1Label;
            newTpoData.append(subPeriod1Data);

            QMap<QString, QVariant> subPeriod2Data = dataPoint;
            subPeriod2Data["time_index"] = timeIndex + 1; // Second sub-period gets the next index
            subPeriod2Data["subperiod_label"] = subPeriod2Label;
            newTpoData.append(subPeriod2Data);
        } else if (current_time_index > timeIndex) {
            // Adjust time index for data points that come after the split point
            newDataPoint["time_index"] = current_time_index + 1;
            newTpoData.append(newDataPoint);
        } else {
            // Keep data points that come before the split point unchanged
            newTpoData.append(newDataPoint);
        }
    }

    // Update the time labels to include the new sub-periods
    QStringList newTimeLabels = m_timeLabels;
    newTimeLabels.insert(timeIndex, subPeriod1Label);  // Insert first sub-period at original position
    newTimeLabels.insert(timeIndex + 1, subPeriod2Label);  // Insert second sub-period after first
    newTimeLabels.removeAt(timeIndex + 2);  // Remove the original time label that was shifted right

    // Update the internal data
    m_tpoData = newTpoData;
    m_timeLabels = newTimeLabels;

    // Update the TPO engine with the new data
    m_tpoEngine.clear();
    for (const auto &dataPoint : m_tpoData) {
        if (dataPoint.contains("price") && dataPoint.contains("time_index")) {
            double price = dataPoint["price"].toDouble();
            int timeIndex = dataPoint["time_index"].toInt();

            // Create a mock timestamp based on time index (in a real implementation, you'd have actual timestamps)
            auto timestamp = std::chrono::system_clock::now() + std::chrono::minutes(timeIndex * 15); // Using 15 min intervals now

            // Create a PriceTick and process it
            PriceTick tick;
            tick.timestamp = timestamp;
            tick.price = price;
            tick.volume = 1.0; // Default volume for TPO counting

            m_tpoEngine.process_tick(tick);
        }
    }

    // Recalculate POC, Value Area, and Single Prints
    const TPOProfile& profile = m_tpoEngine.get_tpo_profile();
    m_cached_poc = profile.get_poc();
    auto va = profile.get_value_area(70.0); // 70% of TPOs
    m_cached_va_low = va.first;
    m_cached_va_high = va.second;
    m_cached_single_prints = profile.get_single_print_levels(); // Update cached single prints
    m_values_cached = true;

    // Clear selection
    m_hasSelection = false;
    m_selectedPriceIndex = -1;
    m_selectedTimeIndex = -1;

    // Trigger a repaint
    update();
}

// Factory function to create the TPO profile panel
extern "C" QWidget* createTPOProfilePanel(QWidget *parent)
{
    return new TPOProfilePanel(parent);
}