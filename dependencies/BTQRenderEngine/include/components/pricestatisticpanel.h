#ifndef PRICESTATISTICPANEL_H
#define PRICESTATISTICPANEL_H

#include <QTableWidget>
#include <QScrollBar>
#include <QWheelEvent>
#include <QVector>
#include <QString>

// Structure to hold price statistics data
struct PriceStatData {
    QString symbol;
    double lastPrice;
    double change;
    double changePercent;
    qint64 volume;
    double high;
    double low;
    double open;
    double close;
    
    // Constructor
    PriceStatData(const QString &sym = "", 
                  double last = 0.0, 
                  double chg = 0.0, 
                  double chgPct = 0.0, 
                  qint64 vol = 0,
                  double h = 0.0,
                  double l = 0.0,
                  double o = 0.0,
                  double c = 0.0)
        : symbol(sym), lastPrice(last), change(chg), changePercent(chgPct), 
          volume(vol), high(h), low(l), open(o), close(c) {}
};

class PriceStatisticPanel : public QTableWidget
{
    Q_OBJECT

public:
    explicit PriceStatisticPanel(QWidget *parent = nullptr);
    
    // Public methods for managing data
    void setData(const QVector<PriceStatData> &data);
    void addData(const PriceStatData &data);
    void updateData(int index, const PriceStatData &data);
    void clearAllData();
    QVector<PriceStatData> getData() const;
    int getTotalRowCount() const;

protected:
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private slots:
    void onScrollValueChanged(int value);

private:
    void initializeUI();
    void setupHeaders();
    void connectScrollSignals();
    void updateVisibleRows();
    void updateScrollbarGeometry();
    void refreshVisibleData();

private:
    QVector<PriceStatData> m_data;
    QScrollBar *m_virtualScrollBar;
    int m_rowHeight;
    int m_visibleRows;
    int m_totalRows;
    int m_firstVisibleRow;
};

#endif // PRICESTATISTICPANEL_H