#ifndef DIALOG1_H
#define DIALOG1_H

#include "ui_dialog1.h"

class Dialog1 : public QDialog, private Ui::Dialog1
{
	Q_OBJECT
	
public:
	explicit Dialog1(QWidget *parent = nullptr);
	~Dialog1();
};

#endif // DIALOG1_H
