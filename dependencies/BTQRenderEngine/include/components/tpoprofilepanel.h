#ifndef TPOPROFILEPANEL_H
#define TPOPROFILEPANEL_H

#include <QWidget>
#include <QVector>
#include <QMap>
#include <QStringList>
#include <QColor>
#include <QVariant>

// Include TPO engine for Value Area and POC calculations
#include "analytics/tpoengine.h"

/**
 * @brief TPO Profile Panel implementing Quantower TPO layout
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

// Factory function to create the TPO profile panel
extern "C" QWidget* createTPOProfilePanel(QWidget *parent = nullptr);

#endif // TPOPROFILEPANEL_H