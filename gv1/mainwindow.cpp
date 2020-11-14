#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QPixmap>
#include <QDebug>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QGraphicsView>
#include <QTransform>
#define AREAX 400000 //領域サイズ 幅の半分
#define AREAY 400000//領域サイズ  高さの半分
#define AWIDTH (AREAX*2.0)  //sceneのサイズ= 領域サイズ 幅
#define AHEIGHT (AREAY*2.0)  //sceneのサイズ= 領域サイズ 高さ
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	m_scene = new QGraphicsScene();
	m_scene->setSceneRect(-AREAX,-AREAY,AREAX*2,AREAY*2);
	ui->gv1->setScene(m_scene);
	//ui->gv1->scale(AWIDTH/ui->gv1->width(), AHEIGHT/ui->gv1->height());
	//ui->gv1->scale(ui->gv1->width()/AWIDTH, ui->gv1->height()/AHEIGHT);
	m_item = nullptr;
	qDebug() << "1 w,h"<< m_scene->width() << m_scene->height();
	qDebug() << "1"<< "sceneRect" << m_scene->sceneRect();
    //QPixmap pmap(":/blackrect.png");
	m_pixmap.load(":/blackrect.png");
	//qDebug() << pmap.rect();
	//m_item = m_scene->addPixmap(pmap);
	//ui->gv1->fitInView(m_item);
	//ui->gv1->show();
	//ui->gv1->update();
	ui->gv1->installEventFilter(this);
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::showEvent(QShowEvent *evt)
{
	//ui->gv1->fitInView(0,0,750,500,Qt::KeepAspectRatioByExpanding);
	//ui->gv1->fitInView(m_item);
	
}

void MainWindow::resizeEvent(QResizeEvent *evt)
{
	
//	ui->gv1->fitInView(m_item);
}


void MainWindow::on_pushButton_clicked()
{
#if 0
	if(m_item){
		m_scene->removeItem(m_item);
	}
	m_scene->clear();
#endif
    //QPixmap pmap(":/cube.png");
	qDebug() << m_pixmap.rect();
	m_item = m_scene->addPixmap(m_pixmap);
	
	//m_item->setPixmap(pmap);
	//m_item = m_scene->addPixmap(pmap);
	qDebug() << "pb1 boundingRect" << m_item->boundingRect();
	qDebug() << "pb1 w,h"<< m_scene->width() << m_scene->height();
	qDebug() << "pb1 sceneRect" << m_scene->sceneRect();
	//ui->gv1->fitInView(m_item);
	QPen pen;
	QBrush brush;
	pen.setColor(Qt::blue);
	pen.setStyle(Qt::PenStyle::SolidLine);
	pen.setWidth(0);
	brush.setColor(QColor(255,0,0,64));
	brush.setStyle(Qt::SolidPattern);
	m_scene->addRect(-AREAX+1, -AREAY+1, AWIDTH-1, AHEIGHT-1, pen, brush);
	
	
}

void MainWindow::on_pushButton_2_clicked()
{
	if(m_item){
	ui->gv1->fitInView(0,0,750,500,Qt::KeepAspectRatioByExpanding);
	m_scene->clear();
	//	m_scene->removeItem(m_item);
	//ui->gv1->fitInView(0,0,750,500,Qt::KeepAspectRatioByExpanding);
	}
	m_scene->setSceneRect(0,0,0,0);
	qDebug() << "w,h"<< m_scene->width() << m_scene->height();
    QPixmap pmap(":/blackrect.png");
	qDebug() << pmap.rect();
	m_item = m_scene->addPixmap(pmap);
	qDebug() << "boundingRect" << m_item ->boundingRect();
	qDebug() << "w,h"<< m_scene->width() << m_scene->height();
	qDebug() << "sceneRect" << m_scene->sceneRect();
	//ui->gv1->fitInView(ittem);
	ui->gv1->hide();
	ui->gv1->show ();
	//ui->gv1->fitInView(0,0,750,500,Qt::KeepAspectRatioByExpanding);
    
}

void MainWindow::on_pushButton_3_clicked()
{
    close();
}

void MainWindow::on_pushButton_move_clicked()
{
    
	//画像内のオフセット  画像内の原点を画像内の座標系で移動する
	m_item->setOffset(ui->spinBoxx->value(), ui->spinBoxy->value());
}

void MainWindow::on_pushButton_4_clicked()
{
	QTransform trans;
	trans.scale(ui->spinBoxscalex->value(), ui->spinBoxscaley->value() );
	m_item->setTransform(trans);
	
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
  QEvent::Type type = event->type();	
  if(obj == ui->gv1){
	  if(type == QEvent::Resize){
  		qDebug() << type << event << obj ;
		QString str;
		str.sprintf("gv= %d,%d,%d,%d\npixmap=%d,%d\nrect=%f,%f", 
			ui->gv1->viewport()->width()
			, ui->gv1->viewport()->height()
			,ui->gv1->width()
			, ui->gv1->height()
			,m_pixmap.size().width()
			, m_pixmap.size().height()
					,m_scene->width()
					,m_scene->height()
		) ;
		qDebug() << str;
		ui->label_sz ->setText(str);
		QTransform trans;//座標変換 単位行列　変換なしの行列
		ui->gv1->scale(ui->gv1->width()/AWIDTH, ui->gv1->height()/AHEIGHT);
		qDebug() << "gv1transform 1"<<ui->gv1->transform();
		ui->gv1->setTransform(trans);
		ui->gv1->scale(ui->gv1->width()/AWIDTH, ui->gv1->height()/AHEIGHT);
		qDebug() << "gv1transform 2"<<ui->gv1->transform();
		qreal sx = AWIDTH/(float)ui->gv1->width();
		qreal sy = AHEIGHT/(float)ui->gv1->height();
		ui->gv1->scale(sx,sy );
		qDebug() << "gv1transform3 "<<ui->gv1->transform() << "sxy" << sx << sy;
		ui->gv1->scale(sx,sy );
		qDebug() << "gv1transform4 "<<ui->gv1->transform() << "sxy" << sx << sy;
		//sceneとgraphicsviewのスケール設定  sceneが大きいので、縮小スケールになる
		ui->gv1->setTransform(trans);//スケールを戻す
		ui->gv1->scale(1/sx,1/sy );//sceneをviewいっぱいに表示する縮小
		qDebug() << "gv1transform5 "<<ui->gv1->transform() << "sxy" << sx << sy;
	  }
  }
  return false;
}

void MainWindow::on_pbsetpos_clicked()
{
 //画像原点(左上)の位置をsceneの座標で指定   
	m_item->setPos(ui->spinposx->value(), ui->spinposy->value());
}

//fit表示
void MainWindow::on_pushButtonfit_clicked()
{
	//表示位置は、sceneの左上にする　そこを画像原点にする  
	QRectF rect = m_scene->sceneRect();//scene領域 中央が0になっている
	m_item->setPos(rect.topLeft());// 画像の原点である左上をsceneの左上の位置にする
	
	//画像をsceneのrectと同じ大きさになるように拡大 縦横、それぞれで拡大率を計算
	QSizeF szpix = m_pixmap.size();//画像サイズ
	//scene.size / pixmap.size
	//QSizeFでは計算ができないので、縦横個別に計算
	QSizeF scale(rect.width()/ szpix.width(), rect.height()/szpix.height());
	QTransform trans;//座標変換  単位行列
	trans.scale(scale.width(),scale.height());//拡大する 対角成分が比率
	qDebug() << "rect"<<rect << "szpix"<<szpix << "scale"<<scale <<"trans"<<trans;
	m_item->setTransform(trans);//座標変換行列を設定
	
	//sceneとgraphicsviewのスケール設定  sceneが大きいので、縮小スケールになる
	QTransform trans1;//座標変換  単位行列
	qreal sx = (float)ui->gv1->width()/AWIDTH;
	qreal sy = (float)ui->gv1->height()/AHEIGHT;
	trans1.scale(sx, sy);
	ui->gv1->setTransform(trans1);//スケールを戻す
	
}
