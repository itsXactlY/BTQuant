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

// Forward declarations or includes for TPO engine
#include "analytics/tpoengine.h"

/**
 * @brief TPO Profile Panel widget implementing Quantower TPO layout
 * 
 * This panel displays Time Price Opportunities (TPO) in a traditional format:
 * - Vertical price scale on the left side
 * - Horizontal letter blocks representing time periods
 */
class TPOProfilePanel : public QWidget
{
    Q_OBJECT

public:
    explicit TPOProfilePanel(QWidget *parent = nullptr);
    ~TPOProfilePanel();

    // Public interface methods
    void setData(const QVector<QMap<QString, QVariant>> &tpoData);
    void setPriceRange(double minPrice, double maxPrice, double step = 0.25);
    void setTimePeriods(const QStringList &timeLabels);
    void setLetterColors(const QMap<QChar, QColor> &colors);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void updateDisplay();

private:
    void drawPriceScale(QPainter &painter);
    void drawLetterBlocks(QPainter &painter);
    void drawGridLines(QPainter &painter);
    QRect getBlockRect(int priceIndex, int timeIndex) const;
    int getPriceIndex(double price) const;
    double getPriceAt(int index) const;

    // Data members
    QVector<QMap<QString, QVariant>> m_tpoData;  // TPO data storage
    double m_minPrice;
    double m_maxPrice;
    double m_priceStep;
    QStringList m_timeLabels;
    QMap<QChar, QColor> m_letterColors;
    
    // Layout dimensions
    int m_priceScaleWidth;
    int m_topMargin;
    int m_bottomMargin;
    int m_leftMargin;
    int m_rightMargin;
    int m_blockWidth;
    int m_blockHeight;
    
    // Display properties
    bool m_needsUpdate;
};

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
    
    // Note: updateDisplay is a private slot that triggers a repaint when data changes
    // The connection would normally be made elsewhere when needed
}

// Destructor implementation
TPOProfilePanel::~TPOProfilePanel()
{
}

void TPOProfilePanel::setData(const QVector<QMap<QString, QVariant>> &tpoData)
{
    m_tpoData = tpoData;
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
    
    for (int i = 0; i < priceLevels; ++i) {
        double price = m_maxPrice - (i * m_priceStep);
        
        // Calculate position for this price level
        int y = m_topMargin + (i * m_blockHeight);
        
        // Draw price label
        QString priceStr = QString::number(price, 'f', 2);
        QRect labelRect(m_leftMargin - m_priceScaleWidth + 5, y, 
                        m_priceScaleWidth - 10, m_blockHeight);
        
        painter.drawText(labelRect, Qt::AlignLeft | Qt::AlignVCenter, priceStr);
        
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

// Factory function to create the TPO profile panel
extern "C" QWidget* createTPOProfilePanel(QWidget *parent)
{
    return new TPOProfilePanel(parent);
}