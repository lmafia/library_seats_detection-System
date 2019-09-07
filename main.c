#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv.hpp>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/shm.h>

#define SERVER "114.215.65.250"
#define PORT 4508
#define BUFF_SIZE 80
#define bufferLen 19
#define lenStd 11
using namespace cv;
using namespace std;

Mat moveDetect(Mat temp_0, Mat temp_1, Mat temp_2, Mat temp_3, Mat frame);
Mat hairColorDetect(Mat src);
Mat skinDetect(Mat &img);
Mat RGB_detect(Mat& img);
int otsuThresh(const Mat src);
double dutycycle(Mat src);
void RemoveSmallRegion(Mat Src, Mat Dst,int AreaLimit, int CheckMode, int NeihborMode);
void fillHole(const Mat src, Mat &dst);
double roiDeal(Mat temp_0, Mat temp_1, Mat temp_2, Mat temp_3, Mat src, int *i, Rect roiSize, int d);
void persDetect(double index ,double thres);
static void ContrastAndBright(int, void *);

int sock_cli = socket(AF_INET,SOCK_STREAM, 0);
Mat frame, frame1, src;                //定义Mat变量，用来存储每一帧
int g_nContrastValue = 100;
int g_nBrightValue = 100;





int main()
{



    ///定义sockaddr_in
    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);  ///服务器端口
    servaddr.sin_addr.s_addr = inet_addr(SERVER);  ///服务器ip
    inet_pton(AF_INET, SERVER, &servaddr.sin_addr);
    char data[]={"KEY:luobin21+woaizhutou"};
    if (connect(sock_cli, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
          exit(1);
    }
    send(sock_cli, data, strlen(data),0);

    VideoCapture cap;
    cap.open(0);             //读取摄像头
    if(!cap.isOpened())        //检查视频/摄像头打开是否成功
         return 0;
    Mat frame, frame1;                 //定义Mat变量，用来存储每一帧
    Mat temp;
    Mat temp_0, temp_1, temp_2, temp_3;
    Mat result, nor_temp;
    namedWindow("【轨迹条】");

    createTrackbar("对比度：", "【轨迹条】",&g_nContrastValue,300,ContrastAndBright );
    createTrackbar("亮   度：","【轨迹条】",&g_nBrightValue,300,ContrastAndBright );

    //循环显示每一帧
    int i=0;
    while(1)
    {
        double index_1[bufferLen]={0}, index_2[bufferLen]={0}, index_3[bufferLen]={0}, index_4[bufferLen]={0};
        cap >> src;//读帧进frame
        frame  = src.clone();

		if (frame.empty())//对帧进行异常检测
		{
			cout << "frame is empty!" << endl;
			break;
		}

        ContrastAndBright(g_nContrastValue,0);
        ContrastAndBright(g_nBrightValue,0);

		Rect roi1(20,40,120,80);                     /*设置ROI区域*/
		Rect roi2(180,40,120,80);
		Rect roi3(340,40,120,80);                     /*设置ROI区域*/
		Rect roi4(500,40,120,80);

        rectangle(src, roi1, Scalar(0,255,0), 5);  /*在原图中把ROI区域用矩形标出*/
		rectangle(src, roi2, Scalar(0,255,0), 5);
		rectangle(src, roi3, Scalar(0,255,0), 5);
		rectangle(src, roi4, Scalar(0,255,0), 5);

        imshow("frame", src);

if (i >= 5)                        /*将ROI区域的结果进一步的确定，避免行人路过*/
{

            int flag_1 = 0, flag_2 = 0, flag_3 = 0, flag_4 = 0;
            for(int j = 0; j<bufferLen; j++)
            {
//                cap >> frame;
                index_1[j] = roiDeal(temp_0, temp_1, temp_2, temp_3, frame, &i, roi1,1);
                index_2[j] = roiDeal(temp_0, temp_1, temp_2, temp_3, frame, &i, roi2,2);
                index_3[j] = roiDeal(temp_0, temp_1, temp_2, temp_3, frame, &i, roi3,3);
                index_4[j] = roiDeal(temp_0, temp_1, temp_2, temp_3, frame, &i, roi4,4);
            }

            for(int k = 0; k<bufferLen; k++)
            {
                if(index_1[k] > 0.1)
                    flag_1++;
                if(index_2[k] > 0.1)
                    flag_2++;
                if(index_3[k] > 0.1)
                    flag_3++;
                if(index_4[k] > 0.1)
                    flag_4++;
            }
                if(flag_1>= lenStd)
                {cout << "位置1有人" <<endl;
                char camera_1_1[]={"DATA:0101+1"};
                send(sock_cli, camera_1_1, strlen(camera_1_1), 0);
                memset(camera_1_1, 0, strlen(camera_1_1));
                }
                else
                {
                char camera_1_0[]={"DATA:0101+0"};
                send(sock_cli, camera_1_0, strlen(camera_1_0), 0);
                memset(camera_1_0, 0, strlen(camera_1_0));
                }

                 if(flag_2>= lenStd)
                {cout << "位置2有人" <<endl;

                }
                 if(flag_3>= lenStd)
                {cout << "位置3有人" <<endl;

                }
                 if(flag_4>= lenStd)
                {cout << "位置4有人" <<endl;

                }


}



		i++;
        temp_3 = frame.clone();
		temp_2 = temp_3.clone();
		temp_1 = temp_2.clone();
		temp_0 = temp_1.clone();
		if (waitKey(1000) == 27)     //  帧率的设置
		{
			cout << "ESC退出!" << endl;
			break;
		}
    }
    cap.release();            //关闭视频流文件
    return 0;

    }

static void ContrastAndBright(int, void *) // trackber回调函数
{

       for(int y = 0; y < src.rows; y++ )
       {
              for(int x = 0; x < src.cols; x++ )
              {
                     for(int c = 0; c < 3; c++ )
                     {
                        frame.at<Vec3b>(y,x)[c]= saturate_cast<uchar>( (g_nContrastValue*0.01)*(src.at<Vec3b>(y,x)[c] ) + g_nBrightValue-100 );
                     }
              }
       }

       //显示图像

}

double roiDeal(Mat temp_0, Mat temp_1, Mat temp_2, Mat temp_3, Mat src, int *i, Rect roiSize, int d)
{
    Mat temp;
    Mat result, nor_temp;
    temp_0 = temp_0(roiSize);
    temp_1 = temp_1(roiSize);
    temp_2 = temp_2(roiSize);
    temp_3 = temp_3(roiSize);
    src = src(roiSize);

        if(*i <= 5)
            result = moveDetect(src, src, src, src ,src);
		else
            result = moveDetect(temp_0, temp_1, temp_2, temp_3, src);
//		  imshow("result", result);
        Mat src_temp;
        src_temp = src;
		temp = skinDetect(src_temp);
		RemoveSmallRegion(temp,temp,200,1,1);
		fillHole(temp,temp);
//       imshow("肤色检测", temp);
//
//        nor_temp = hairColorDetect(src);
//        bitwise_not(hairColorDetect(src),nor_temp);
//        fillHole(nor_temp,nor_temp);
//		RemoveSmallRegion(nor_temp,nor_temp,100,1,1);
//        imshow("发色检测", nor_temp);

        Mat or_temp, detect;
//        bitwise_or(nor_temp, temp, or_temp);
//        bitwise_or(or_temp, result, detect);
        bitwise_or(temp, result, detect);
		char str[20] = {0};
		char str_1[20] = {0};
		sprintf(str,"detect%d",d);
		sprintf(str_1,"位置%d的占空比：",d);
		namedWindow(str);
		imshow(str, detect);

        double dutyratio = dutycycle(detect);
//        cout << str_1 << dutyratio << endl;
        return dutyratio;
}


//肤色检测
Mat skinDetect(Mat &src)
{
    Mat img = src.clone();
    medianBlur(img, img, 3);
    Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);
    ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
    Mat ycrcb_image;
    Mat output_mask = Mat::zeros(img.size(), CV_8UC1);
    cvtColor(img, ycrcb_image, CV_BGR2YCrCb);
    for (int i = 0; i < img.cols; i++)
        for (int j = 0; j < img.rows; j++)
        {
            Vec3b ycrcb = ycrcb_image.at<Vec3b>(j, i);
            if (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0)
                output_mask.at<uchar>(j, i) = 255;
        }
    Mat detect;
    img.copyTo(detect,output_mask);
    cvtColor(detect,detect,CV_BGR2GRAY);
    threshold(detect,detect,10,255,CV_THRESH_BINARY);
    return detect;
}

//OTSU自适应阈值
int otsuThresh(const Mat src)
{
    Mat img = src;
    int c = img.cols; //图像列数
    int r = img.rows; //图像行数
    int T = 0; //阈值
    uchar* data = img.data; //数据指针
    int ftNum = 0; //前景像素个数
    int bgNum = 0; //背景像素个数
    int N = c*r; //总像素个数
    int ftSum = 0; //前景总灰度值
    int bgSum = 0; //背景总灰度值
    int graySum = 0;
    double w0 = 0; //前景像素个数占比
    double w1 = 0; //背景像素个数占比
    double u0 = 0; //前景平均灰度
    double u1 = 0; //背景平均灰度
    double Histogram[256] = {0}; //灰度直方图
    double temp = 0; //临时类间方差
    double g = 0; //类间方差

    //灰度直方图
    for(int i = 0; i < r ; i ++) {
        for(int j = 0; j <c; j ++) {
            Histogram[img.at<uchar>(i,j)]++;
        }
    }
    //求总灰度值
    for(int i = 0; i < 256; i ++) {
        graySum += Histogram[i]*i;
    }

    for(int i = 0; i < 256; i ++) {
        ftNum += Histogram[i];  //阈值为i时前景个数
        bgNum = N - ftNum;      //阈值为i时背景个数
        w0 = (double)ftNum/N; //前景像素占总数比
        w1 = (double)bgNum/N; //背景像素占总数比
        if(ftNum == 0) continue;
        if(bgNum == 0) break;
        //前景平均灰度
        ftSum += i*Histogram[i];
        u0 = ftSum/ftNum;
        //背景平均灰度
        bgSum = graySum - ftSum;
        u1 = bgSum/bgNum;

        g = w0*w1*(u0-u1)*(u0-u1);
        if(g > temp) {
            temp = g;
            T = i;
        }
    }
// cout <<"结果阈值："<<T<<endl;
 return T;
}

//运动检测
Mat moveDetect(Mat temp_0, Mat temp_1, Mat temp_2, Mat temp_3, Mat frame)
{
    //转为灰度图
    Mat result = frame.clone();
    Mat result_1 = frame.clone();
	cvtColor(temp_0, temp_0, CV_BGR2GRAY);
    cvtColor(temp_1, temp_1, CV_BGR2GRAY);
    cvtColor(temp_2, temp_2, CV_BGR2GRAY);
    cvtColor(temp_3, temp_3, CV_BGR2GRAY);
    cvtColor(frame , frame,  CV_BGR2GRAY);
    //均值滤波
    blur(temp_0, temp_0, Size(3,3));
    blur(temp_1, temp_1, Size(3,3));
    blur(temp_2, temp_2, Size(3,3));
    blur(temp_3, temp_3, Size(3,3));
    blur(frame , frame , Size(3,3));
    //灰度均值化
    equalizeHist(temp_0, temp_0);
    equalizeHist(temp_1, temp_1);
    equalizeHist(temp_2, temp_2);
    equalizeHist(temp_3, temp_3);
    equalizeHist(frame , frame );
    //做差，求值
    Mat D_00, D_01, D_02, D_03;
    absdiff(temp_0, frame, D_00);
    absdiff(temp_1, frame, D_01);
    absdiff(temp_2, frame, D_02);
    absdiff(temp_3, frame, D_03);
    //膨胀并且二值化
    Mat kernel_dilate = getStructuringElement(MORPH_CROSS, Size(3, 3));

    dilate(D_00, D_00, kernel_dilate);
    dilate(D_01, D_01, kernel_dilate);
    dilate(D_02, D_02, kernel_dilate);
    dilate(D_03, D_03, kernel_dilate);

    threshold(D_00, D_00, otsuThresh(D_00)+60, 255, CV_THRESH_BINARY);
    threshold(D_01, D_01, otsuThresh(D_01)+60, 255, CV_THRESH_BINARY);
    threshold(D_02, D_02, otsuThresh(D_02)+60, 255, CV_THRESH_BINARY);
    threshold(D_03, D_03, otsuThresh(D_03)+60, 255, CV_THRESH_BINARY);
    //D0与D1 或 D2与D3 输出给result
    Mat D_1, D_2;
    bitwise_and(D_00, D_01, D_1);
    bitwise_and(D_02, D_03, D_2);
    bitwise_or(D_1, D_2, result);
    threshold(result, result, otsuThresh(result)+65, 255, CV_THRESH_BINARY);
    dilate(result, result, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)));
    fillHole(result,result);

    return result;
////    imshow("检测图",result);
//    //查找轮廓并绘制轮廓
//	vector<vector<Point>> contours;
//	findContours(result, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
////	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓
//    fillHole(result,result);
//	//查找正外接矩形
//	vector<Rect> boundRect(contours.size());
//	for (int i = 0; i < contours.size(); i++)
//	{
//		boundRect[i] = boundingRect(contours[i]);
//		rectangle(result_1, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形
//	}


}

//去除最小孔洞.
void fillHole(const Mat src, Mat &dst)
{
	Size m_Size = src.size();
	Mat temimage = Mat::zeros(m_Size.height + 2, m_Size.width + 2, src.type());//延展图像
	src.copyTo(temimage(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
	floodFill(temimage, Point(0, 0), Scalar(255));
	Mat cutImg;//裁剪延展的图像
	temimage(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
	dst = src | (~cutImg);
}

//去除最小孔洞，最小连通区域。
void RemoveSmallRegion(Mat Src, Mat Dst,int AreaLimit, int CheckMode, int NeihborMode)
{

	int RemoveCount = 0;
	//新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
	//初始化的图像全部为0，未检查
	Mat PointLabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)//去除小连通区域的白色点
	{
//		cout << "去除小连通域.";
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) < 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//将背景黑色点标记为合格，像素为3
				}
			}
		}
	}

	else//去除孔洞，黑色点像素
	{
//		cout << "去除孔洞";
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) > 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//如果原图是白色区域，标记为合格，像素为3
				}
			}
		}
	}
//将邻域压进容器
	vector<Point2i>NeihborPos;
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{
//		cout << "Neighbor mode: 8邻域." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
//	else cout << "Neighbor mode: 4邻域." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			if (PointLabel.at<uchar>(i, j) == 0)//标签图像像素点为0，表示还未检查的不合格点
			{   //开始检查
				vector<Point2i>GrowBuffer;//记录检查像素点的个数
				GrowBuffer.push_back(Point2i(j, i));
				PointLabel.at<uchar>(i, j) = 1;//标记为正在检查
				int CheckResult = 0;
 				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					for (int q = 0; q < NeihborCount; q++)
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //防止越界
						{
							if (PointLabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer
								PointLabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查
							}
						}
					}
				}
				if (GrowBuffer.size()>AreaLimit) //判断结果（是否超出限定的大小），1为未超出，2为超出
					CheckResult = 2;
				else
				{
					CheckResult = 1;
					RemoveCount++;//记录有多少区域被去除
				}
				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					PointLabel.at<uchar>(CurrY,CurrX)+=CheckResult;//标记不合格的像素点，像素值为2
				}
				//********结束该点处的检查**********
			}
		}
	}
	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域
	for (int i = 0; i < Src.rows; ++i)
	{
		for (int j = 0; j < Src.cols; ++j)
		{
			if (PointLabel.at<uchar>(i,j)==2)
			{
				Dst.at<uchar>(i, j) = CheckMode;
			}
			else if (PointLabel.at<uchar>(i, j) == 3)
			{
				Dst.at<uchar>(i, j) = Src.at<uchar>(i, j);

			}
		}
	}
//	cout << RemoveCount << " objects removed." << endl;
}

//计算像素占空比 (输入二值图）
double  dutycycle(Mat src)
{
  int N = src.cols* src.rows;
  int B = countNonZero(src);
  double R = (double)B/ N;
  return R;
}

//发色检测
Mat hairColorDetect(Mat src)
{
    Mat dst;
    cvtColor(src, dst , CV_BGR2HSV);
    vector<Mat> channels;
    Mat hChannel;
    Mat sChannel;
    Mat vChannel;
    medianBlur(dst,dst,5);
    split(dst,channels);//分离色彩通道
    hChannel = channels.at(0);
    sChannel = channels.at(1);
    vChannel = channels.at(2);
    threshold(vChannel,vChannel,40,255,CV_THRESH_BINARY);
    return vChannel;
}
