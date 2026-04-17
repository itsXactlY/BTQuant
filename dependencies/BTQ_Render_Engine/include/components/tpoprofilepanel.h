#ifndef TPOPROFILEPANEL_H
#define TPOPROFILEPANEL_H

#include <QWidget>
#include <QVector>
#include <QMap>
#include <QStringList>
#include <QColor>
#include <QVariant>
#include <QMouseEvent>
#include <QContextMenuEvent>
#include <QMenu>
#include <QAction>

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
    void mousePressEvent(QMouseEvent *event) override;
    void contextMenuEvent(QContextMenuEvent *event) override;

private slots:
    void updateDisplay();
    void splitProfileAction();

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

    // TPO engine for advanced calculations
    TPOEngine m_tpoEngine;

    // Cached POC and Value Area values
    double m_cached_poc;
    double m_cached_va_low;
    double m_cached_va_high;
    std::vector<double> m_cached_single_prints;  // Cache for single print levels
    bool m_values_cached;

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

    // Selected TPO bar information for context menu
    int m_selectedPriceIndex;
    int m_selectedTimeIndex;
    bool m_hasSelection;

    // Context menu
    QMenu* m_contextMenu;
    QAction* m_splitProfileAction;
};

// Factory function to create the TPO profile panel
extern "C" QWidget* createTPOProfilePanel(QWidget *parent = nullptr);

#endif // TPOPROFILEPANEL_H