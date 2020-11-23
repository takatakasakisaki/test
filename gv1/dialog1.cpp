#include "dialog1.h"
#include <QDebug>

Dialog1::Dialog1(QWidget *parent) :
	QDialog(parent)
{
	setupUi(this);
	
	qDebug() << dialoggv->viewport()->width() << dialoggv->viewport()->height();
}

Dialog1::~Dialog1()
{
	
}
