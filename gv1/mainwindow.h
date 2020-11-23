#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QRubberBand>
#include "dialog1.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
	void showEvent(QShowEvent*evt);
	void resizeEvent(QResizeEvent*evt);
	QGraphicsScene *m_scene;
	QGraphicsPixmapItem *m_item;
	bool gv1_eventFilter(QObject *obj, QEvent *event);
	bool gv1_Resize(QObject *obj, QEvent *event);
private slots:
    void on_pushButton_clicked();
    
	void on_pushButton_2_clicked();
	
	void on_pushButton_3_clicked();
	
	void on_pushButton_move_clicked();
	
	void on_pushButton_4_clicked();
	
	void on_pbsetpos_clicked();
	
	void on_pushButtonfit_clicked();
	
	void on_pbLoadBlack_clicked();
	
	void on_pbfitheight_clicked();
	
private:
    Ui::MainWindow *ui;
	QRubberBand *m_rubberBand;
	QPoint m_origin ;
	Dialog1 *m_dialog1;
protected:
    bool eventFilter(QObject* obj, QEvent* event);	
	QPixmap m_pixmap;
};
#endif // MAINWINDOW_H
