#include "myview.h"
#include <QDebug>

myview::myview(QWidget *parent)
	:QGraphicsView(parent )
{
	
	qDebug() << "myview "<<viewport()->width() << viewport()->height();
}

void myview::resizeEvent(QResizeEvent *event)
{
	qDebug() << "resizeEvent "<< event << viewport()->width() << viewport()->height();
	QGraphicsView::resizeEvent(event);
}

void myview::showEvent(QShowEvent *event)
{
	qDebug() << "showevent "<< event << viewport()->width() << viewport()->height();
	QGraphicsView::showEvent(event);
}

void myview::paintEvent(QPaintEvent *event)
{
	
	qDebug() << "paintEvent "<< event << viewport()->width() << viewport()->height();
	QGraphicsView::paintEvent(event);
}

void myview::mouseMoveEvent(QMouseEvent *event)
{
	qDebug() << "mouseMoveEvent" << (QEvent*)event;
	QGraphicsView::mouseMoveEvent(event);
}

void myview::mousePressEvent(QMouseEvent *event)
{
	qDebug() << "mousePressEvent" << (QEvent*)event;
	QGraphicsView::mousePressEvent(event);
}

void myview::mouseReleaseEvent(QMouseEvent *event)
{
	qDebug() << "mouseReleaseEvent" << (QEvent*)event;
	QGraphicsView::mouseReleaseEvent(event);
}
