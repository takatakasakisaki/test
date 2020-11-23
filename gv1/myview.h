#ifndef MYVIEW_H
#define MYVIEW_H

#include <QWidget>
#include <QGraphicsView>

class myview : public QGraphicsView
{
	Q_OBJECT
public:
	myview(QWidget *parent = nullptr);
	void resizeEvent(QResizeEvent *event) override;
	void showEvent(QShowEvent *event) override;
	void paintEvent(QPaintEvent *event) override;
	void mouseMoveEvent(QMouseEvent *event) override;
	void mousePressEvent(QMouseEvent *event) override;
	void mouseReleaseEvent(QMouseEvent *event) override;
};

#endif // MYVIEW_H
