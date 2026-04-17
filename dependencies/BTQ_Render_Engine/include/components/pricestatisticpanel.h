#ifndef PRICESTATISTICPANEL_H
#define PRICESTATISTICPANEL_H

#include <QWidget>
#include <QScrollBar>
#include <QWheelEvent>
#include <QVector>
#include <QString>
#include <QPainter>
#include <QMouseEvent>
#include <QKeyEvent>

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

class PriceStatisticPanel : public QWidget
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
    int getSelectedRow() const;

signals:
    void rowSelected(int rowIndex);
    void headerClicked(int columnIndex);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void onScrollValueChanged(int value);

private:
    void initializeUI();
    void setupHeaders();
    void updateColumnWidths();
    void connectScrollSignals();
    void updateVisibleRows();
    void updateScrollbarGeometry();
    void paintHeaders(QPainter &painter);
    void paintDataRows(QPainter &painter);
    void paintCell(QPainter &painter, const PriceStatData &stat, int row, int yPos);
    void paintSelectionHighlight(QPainter &painter, int visualRow);
    QRect getColumnRect(int column, int row) const;
    int getColumnAtPosition(int x) const;
    void ensureRowVisible(int row);

private:
    QVector<PriceStatData> m_data;
    QScrollBar *m_virtualScrollBar;
    QStringList m_headers;
    QList<int> m_columnWidths;
    int m_columnCount;
    int m_rowHeight;
    int m_visibleRows;
    int m_totalRows;
    int m_firstVisibleRow;
    int m_selectedRow;
};

#endif // PRICESTATISTICPANEL_H