/**
  */
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QPixmap>
#include <QDebug>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QGraphicsView>
#include <QTransform>
#include <QEvent>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QFileDialog>
#include "dialog1.h"
#define AWIDTH (m_pixmap.width())  //sceneのサイズ= 領域サイズ 幅
#define AHEIGHT (m_pixmap.height())  //sceneのサイズ= 領域サイズ 高さ
#define AREAX (-AWIDTH/2) //領域サイズ 幅の半分
#define AREAY (-AHEIGHT/2)//領域サイズ  高さの半分
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	m_scene = nullptr;
	m_item = nullptr;
	auto vp0 = ui->gv1->viewport();
	qDebug() << "gv1->viewport" << vp0->width() << vp0->height();
	
	m_scene = new QGraphicsScene(0,0,1920,1200);
//	m_scene = new QGraphicsScene();
	
	//m_scene->setSceneRect(-AREAX,-AREAY,AREAX*2,AREAY*2);
	ui->gv1->setScene(m_scene);
	vp0 = ui->gv1->viewport();
	qDebug() << "gv1->viewport" << vp0->width() << vp0->height();
	//ui->gv1->scale(AWIDTH/ui->gv1->width(), AHEIGHT/ui->gv1->height());
	//ui->gv1->scale(ui->gv1->width()/AWIDTH, ui->gv1->height()/AHEIGHT);
	//qDebug() << "1 w,h"<< m_scene->width() << m_scene->height();
	//qDebug() << "1"<< "sceneRect" << m_scene->sceneRect();
    //QPixmap pmap(":/blackrect.png");
	//m_pixmap.load(":/blackrect.png");
	m_pixmap.load(":/1920x1200.png");
	//qDebug() << pmap.rect();
	m_item = m_scene->addPixmap(m_pixmap);
	auto vp = ui->gv1->viewport();
	qDebug() << "gv1->viewport" << vp->width() << vp->height();
	ui->gv1->fitInView(m_item);
	vp = ui->gv1->viewport();
	qDebug() << "gv1->viewport" << vp->width() << vp->height();
	//ui->gv1->show();
	//ui->gv1->update();
	ui->gv1->installEventFilter(this);//gv1のイベントを eventFilter に通知する
	ui->gv1->viewport()->installEventFilter(this);//gv1のイベントを eventFilter に通知する
	
	
	//m_dialog1 = new Dialog1(this);
	//m_dialog1->show();
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::showEvent(QShowEvent *evt)
{
	//ui->gv1->fitInView(0,0,750,500,Qt::KeepAspectRatioByExpanding);
	//ui->gv1->fitInView(m_item);
	QMainWindow::showEvent(evt);
	
}

void MainWindow::resizeEvent(QResizeEvent *evt)
{
	
//	ui->gv1->fitInView(m_item);
	QMainWindow::resizeEvent(evt);
}

/**
 * @brief MainWindow::on_pushButton_clicked
 * add pixmap
 */
void MainWindow::on_pushButton_clicked()
{
#if 0
	if(m_item){
		m_scene->removeItem(m_item);
	}
	m_scene->clear();
#endif
	 QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                 "./",
                                                 tr("Images (*.png *.xpm *.jpg)"));
	//m_pixmap.load(":/1920x1200.png");
	if(fileName.isEmpty()){
		return;
	}
	m_pixmap.load(fileName);
	qDebug() << m_pixmap.rect();
	m_item = m_scene->addPixmap(m_pixmap);
	QGraphicsView *gv = ui->gv1;
	
	ui->gv1->fitInView(m_item);
	//m_item->setPixmap(pmap);
	//m_item = m_scene->addPixmap(pmap);
	qDebug() << "pb1 boundingRect" << m_item->boundingRect();
	qDebug() << "pb1 w,h"<< m_scene->width() << m_scene->height();
	qDebug() << "pb1 sceneRect" << m_scene->sceneRect();
	auto viewport = gv->viewport();
	auto rect = viewport->rect();
	qDebug() << rect << "view" << viewport->width() << viewport->height();
	qDebug() << "view transform" << gv->transform();
	qreal sx = rect.width() / (qreal)m_pixmap.width();
	qreal sy = rect.height() / (qreal)m_pixmap.height();
	qreal gsx = gv->width() / (qreal)m_pixmap.width();
	qreal gsy = gv->height() / (qreal)m_pixmap.height();
	qDebug() << "scale=" << sx << sy << "gsx,sy" << gsx << gsy;
	auto offset= m_item->offset();
	auto pos= m_item->pos();
	qDebug() << "offset=" << offset << "pos" << pos << "scale" << m_item->scale();
#if 0
	QPen pen;
	QBrush brush;
	pen.setColor(Qt::blue);
	pen.setStyle(Qt::PenStyle::SolidLine);
	pen.setWidth(0);
	brush.setColor(QColor(255,0,0,64));
	brush.setStyle(Qt::SolidPattern);
	m_scene->addRect(-AREAX+1, -AREAY+1, AWIDTH-1, AHEIGHT-1, pen, brush);
#endif
	
	
}
/**
 * @brief MainWindow::on_pushButton_2_clicked
 * fit w,h
 */
void MainWindow::on_pushButton_2_clicked()
{
#if 0
	if(m_item){
	ui->gv1->fitInView(0,0,750,500,Qt::KeepAspectRatioByExpanding);
	m_scene->clear();
	//	m_scene->removeItem(m_item);
	//ui->gv1->fitInView(0,0,750,500,Qt::KeepAspectRatioByExpanding);
	}
#endif
    //QPixmap pmap(":/blackrect.png");
	//m_scene = new QGraphicsScene();
	
	//m_scene->setSceneRect(0,0,m_pixmap.width(), m_pixmap.height());
	auto sceneRect = m_scene->sceneRect();
	QRectF viewrect = ui->gv1->viewport()->rect();
	qDebug() << "0,0,0,0"<< sceneRect << "w,h"<< m_scene->width() << m_scene->height();
	//m_scene->setSceneRect(0,0,pmap.width(),pmap.height());
	//sceneRect = m_scene->sceneRect();
	qDebug() << sceneRect << "w,h"<< m_scene->width() << m_scene->height();
	qDebug() << m_pixmap.rect();
	//m_item = m_scene->addPixmap(m_pixmap);
	qDebug() << "boundingRect" << m_item ->boundingRect();
	qDebug() << "w,h"<< m_scene->width() << m_scene->height();
	qDebug() << "sceneRect" << m_scene->sceneRect();
	//ui->gv1->setScene(m_scene);
	qreal sx = viewrect.width()  / m_pixmap.width();
	qreal sy = viewrect.height() / m_pixmap.height();
	QTransform trans;
	trans.scale(sx,sy);
	qDebug() << trans;
	ui->gv1->setTransform(trans);
	//ui->gv1->fitInView(m_item);
	//ui->gv1->hide();
	//ui->gv1->show ();
	//ui->gv1->fitInView(0,0,750,500,Qt::KeepAspectRatioByExpanding);
    
}
/**
 * @brief MainWindow::on_pushButton_3_clicked
 * close
 */
void MainWindow::on_pushButton_3_clicked()
{
    close();
}

void MainWindow::on_pushButton_move_clicked()
{
    
	//画像内のオフセット  画像内の原点を画像内の座標系で移動する
	m_item->setOffset(ui->spinBoxx->value(), ui->spinBoxy->value());
}
/**
 * @brief MainWindow::on_pushButton_4_clicked
 * scale
 */
void MainWindow::on_pushButton_4_clicked()
{
	QTransform trans;
	trans.scale(ui->SpinBoxscalex->value(), ui->SpinBoxscaley->value() );
	ui->gv1->setTransform(trans);
	
}
bool MainWindow::gv1_Resize(QObject *obj, QEvent *event)
{
	QResizeEvent *resevt = dynamic_cast<QResizeEvent*>(event);
	qDebug() << "resize" << event << resevt << obj ;
	QString str;
	str = QString::asprintf("gv= vpw=%d,vph=%d,w=%d,h=%d\npixmap=w=%d,h=%d", 
		ui->gv1->viewport()->width()
		, ui->gv1->viewport()->height()
		,ui->gv1->width()
		, ui->gv1->height()
		,m_pixmap.size().width()
		, m_pixmap.size().height()
	) ;
	if(m_scene){
		str += QString::asprintf("\nscene %f,%f",
				 m_scene->width()
				,m_scene->height()
				 );
	}
	qDebug() << str;
	ui->label_sz ->setText(str);
#if 0
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
#endif	
	return true;
	
}
bool MainWindow::gv1_eventFilter(QObject *obj, QEvent *event)
{
	QEvent::Type type = event->type();	
	qDebug() << __func__ << obj << type ;
	bool ret=true;
	switch(type){
	case  QEvent::Resize:
		ret =  gv1_Resize(obj, event);
		break;	
	case  QEvent::MouseButtonPress:
	{
		qDebug() << "MouseButtonPress";
		QMouseEvent *mouseevent = dynamic_cast<QMouseEvent*>(event);
		m_origin = mouseevent->pos();
		if (!m_rubberBand){
			m_rubberBand = new QRubberBand(QRubberBand::Rectangle, this);
		}
		qDebug() << "origin="<<m_origin;
		m_rubberBand->setGeometry(QRect(m_origin, QSize()));
		m_rubberBand->show();
	}
		break;
	case  QEvent::MouseMove:
	{
		QMouseEvent *mouseevent = dynamic_cast<QMouseEvent*>(event);
		qDebug() << "MouseMove" << mouseevent;
		m_rubberBand->setGeometry(QRect(m_origin, mouseevent->pos()).normalized());
	}
		break;
	case  QEvent::MouseButtonRelease:
	{
		qDebug() << "MouseButtonRelease";
		 m_rubberBand->hide();
    // determine selection, for example using QRect::intersects()
    // and QRect::contains().
	
	}
		break;
	default:
		break;
	}
	return ret;
	
}
/**
 * @brief MainWindow::eventFilter
 * 子オブジェクトのイベントを取得したときの処理	
 * @param obj
 * @param event
 * @return 
 */
bool MainWindow::eventFilter(QObject *obj, QEvent *event)
{
  QEvent::Type type = event->type();	
  qDebug() <<"eventFilter" << type << event << obj;
  //if(obj == ui->gv1){
		gv1_eventFilter(obj, event);
  //}
  return false;
}
/**
 * @brief MainWindow::on_pbsetpos_clicked
 * setpos
 */
void MainWindow::on_pbsetpos_clicked()
{
 //画像原点(左上)の位置をsceneの座標で指定   
	m_item->setPos(ui->spinposx->value(), ui->spinposy->value());
}

//fit width表示
void MainWindow::on_pushButtonfit_clicked()
{
	//表示位置は、sceneの左上にする　そこを画像原点にする  
	QRectF rect = m_scene->sceneRect();//scene領域 中央が0になっている
	
	if(!m_item){
		m_item = m_scene->addPixmap(m_pixmap);
	}
	m_item->setPos(rect.topLeft());// 画像の原点である左上をsceneの左上の位置にする
	QRectF viewrect = ui->gv1->viewport()->rect();
	QRectF gvrect = ui->gv1->rect();
	
	//画像をsceneのrectと同じ大きさになるように拡大 縦横、それぞれで拡大率を計算
	QSizeF szpix = m_pixmap.size();//画像サイズ
	//scene.size / pixmap.size
	//QSizeFでは計算ができないので、縦横個別に計算
	QSizeF scale(rect.width()/ szpix.width(), rect.height()/szpix.height());
	QTransform trans;//座標変換  単位行列
	//trans.scale(scale.width(),scale.height());//拡大する 対角成分が比率
	qDebug() << "rect"<<rect << "szpix"<<szpix << "scale"<<scale <<"trans"<<trans;
	m_item->setTransform(trans);//座標変換行列を設定
	
	//sceneとgraphicsviewのスケール設定  sceneが大きいので、縮小スケールになる
	QTransform trans1;//座標変換  単位行列
	//qreal sx = (float)(viewrect.width()-10)/AWIDTH;
	qreal sx = (float)(gvrect.width()-10)/AWIDTH;
	qreal sy = (float)viewrect.height()/AHEIGHT;
	qDebug() << sx << sy;
	trans1.scale(sx, sx);//縦横比は変えない
	ui->gv1->setTransform(trans1);//スケールを戻す
	
}

void MainWindow::on_pbLoadBlack_clicked()
{
	m_pixmap.load(":/1920x1200.png");
	m_item = m_scene->addPixmap(m_pixmap);
	QRectF rect = m_pixmap.rect();
	m_scene->setSceneRect(rect);
	qDebug() << rect;
	QTransform trans1;//座標変換  単位行列
	ui->gv1->setTransform(trans1);//スケールを戻す
    
}
/**
 * @brief MainWindow::on_pbfitheight_clicked
 * fitheight
 */
void MainWindow::on_pbfitheight_clicked()
{
    
	qreal sy0 = (float)(ui->gv1->height()-30)/AHEIGHT;
	qreal sy = (float)ui->gv1->viewport()->height()/AHEIGHT;
	QString str = QString::asprintf("fitheight\nsy0=%f,sy=%f,gv.h=%f,vp.h=%f", 
									sy0,sy,(float)ui->gv1->height(), 
									(float)ui->gv1->viewport()->height()
									);
	qDebug() << str;
	ui->pbfitheight->setText(str);
	QTransform trans;//座標変換  単位行列
	trans.scale(sy0, sy0);
	ui->gv1->setTransform(trans);//スケールを戻す
}
