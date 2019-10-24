// aug_helper.cpp : 定义控制台应用程序的入口点。
//
#include"aug_solver/aug_function.h"

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
#include<string>

#include<boost/program_options.hpp>  
namespace po = boost::program_options;

using namespace std;
using namespace cv;

Mat img;
int opt_bar = 0;
int opt_max_number = 24;
int param_bar = 10;

vector<double> param(1);
vector<Mat> aug_mat(1);

void on_trackbar_share(int value,void *p)
{
    string aug_text = "AUG_CONFIG = ";
    string fanwei = "Range = ";
    string shuoming = "Explanation: ";
    imshow("原图", img);
    switch (opt_bar)
    {
    case 0://Rotate
        param[0] = 1.0 / 100.0*360.0 * param_bar;
        augmentor_png::imRotate(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text+"Rotate("+to_string(param[0])+")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "(---,+++)", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming+"Rotate by one angle (degree)", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 1://FlipUD
        augmentor_png::imFlipUD(img, aug_mat);
        cv::putText(aug_mat[0], aug_text + "FlipUD", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "(NULL)", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Flip up down", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 2://FlipLR
        augmentor_png::imFlipLR(img, aug_mat);
        cv::putText(aug_mat[0], aug_text + "FlipLR", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "(NULL)", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Flip left right", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 3://Flip
        augmentor_png::imFlip(img, aug_mat);
        cv::putText(aug_mat[0], aug_text + "Flip", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "(NULL)", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Random flip", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 4://Crop
        param[0] = 1.0 / 100.0* param_bar;
        param[0] = param[0] < 0.01 ? 0.01 : param[0];
        augmentor_png::imCrop(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "Crop(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "(0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Random crop image by area percentage", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 5://AffineX
        param[0] = 1.0 / 50.0* (param_bar - 50);
        augmentor_png::imAffineX(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "AffineX(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[-1,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "X axis affine", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 6://AffineY
        param[0] = 1.0 / 50.0* (param_bar - 50);
        augmentor_png::imAffineY(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "AffineY(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[-1,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Y axis affine", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 7://AffineY
        param[0] = 1.0 / 50.0* (param_bar - 50);
        augmentor_png::imAffine(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "Affine(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[-1,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "X or Y axis affine", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 8://Noise
        param[0] = 1.0 / 100.0* (param_bar);
        augmentor_png::imAddNoise_gauss(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "Noise(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Add gauss noise", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 9://Hue
        param[0] = 1.0 / 100.0*360.0 * param_bar;
        augmentor_png::imHue(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "Hue(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "(---,+++)", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Hue angle (degree)", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 10://Saturation
        param[0] = 1.0 / 50.0* (param_bar - 50);
        augmentor_png::imSaturation(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "Saturation(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Saturation", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 11://Lightness
        param[0] = 1.0 / 50.0* (param_bar - 50);
        augmentor_png::imLightness(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "Lightness(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Lightness", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 12://PerspectiveUL
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspectiveUL(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "PerspectiveUL(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "UL Perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 13://PerspectiveU
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspectiveU(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "PerspectiveU(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "U  Perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 14://PerspectiveUR
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspectiveUR(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "PerspectiveUR(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "UR Perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 15://PerspectiveL
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspectiveL(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "PerspectiveL(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "L  Perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 16://Perspective
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspective(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "Perspective(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Random perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 17://PerspectiveR
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspectiveR(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "PerspectiveR(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "R  Perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 18://PerspectiveDL
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspectiveR(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "PerspectiveDL(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "DL Perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 19://PerspectiveD
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspectiveD(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "PerspectiveD(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "D  Perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 20://PerspectiveDR
        param[0] = 1.0 / 100.0* param_bar;
        augmentor_png::imPerspectiveDR(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "PerspectiveDR(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,1]", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "DR Perspective", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 21://Distort
        param[0] = 1.0 / 20.0* param_bar;
        augmentor_png::imDistort(img, aug_mat, param);
        cv::putText(aug_mat[0], aug_text + "Distort(" + to_string(param[0]) + ")", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "[0,+++)", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Add distort", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 22://Pyramid
        aug_mat[0].setTo(Scalar(255 / 2, 255 / 2, 255 / 2));
        cv::putText(aug_mat[0], aug_text + "Pyramid(donw,up)", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "(0,+++)", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "Build image pyramid", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    case 23://NotBackup
        aug_mat[0].setTo(Scalar(255 / 2, 255 / 2, 255 / 2));
        cv::putText(aug_mat[0], aug_text + "NotBackup", cv::Point(30, 30), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], fanwei + "(NULL)", cv::Point(30, 60), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        cv::putText(aug_mat[0], shuoming + "A flag means do not keep prestep images", cv::Point(30, 90), 1, 1.5, cv::Scalar(255/2, 0 , 255/2), 2);
        imshow("增广图", aug_mat[0]);
        break;
    }
    return;
}

int main(int argc,char ** argv)
{

    po::options_description desc("aug helper");
    string image_str;

    desc.add_options()
        ("image", po::value<string>(&image_str), "Image for view.");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (boost::program_options::error_with_no_option_name &ex)
    {
        cout << ex.what() << endl;
        //cout << desc << endl;
        return -1;
    }

    if (argc == 1)
    {
        cout << desc << endl;
    }

    img = imread(image_str);
    if (!img.empty())
    {
        double rate = sqrt(750.0*500.0 / img.rows / img.cols);
        resize(img, img, Size(img.cols*rate, img.rows*rate));
    }
    else
    {
        cout << "Use default image for view." << endl;
        img.create(450, 750, CV_8UC3);
        img.setTo(Scalar(255, 255, 255));
        for (int j = 0; j < 450 / 50 + 1; ++j)
        {
            line(img, Point(0, j * 50), Point(750, j * 50), Scalar(0, 0, 0),3);
        }
        for (int j = 0; j < 750 / 50 + 1; ++j)
        {
            line(img, Point(j * 50, 0), Point(j * 50, 450), Scalar(0, 0, 0),3);
        }
        img(Rect(6 * 50 + 3, 3 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(255, 0, 255));
        img(Rect(7 * 50 + 3, 3 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(0, 0, 255));
        img(Rect(8 * 50 + 3, 3 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(0, 255, 255));
        img(Rect(6 * 50 + 3, 4 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(0, 0, 0));
        img(Rect(7 * 50 + 3, 4 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(255, 255, 255));
        img(Rect(8 * 50 + 3, 4 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(255 / 2, 255 / 2, 255 / 2));
        img(Rect(6 * 50 + 3, 5 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(255, 0, 0));
        img(Rect(7 * 50 + 3, 5 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(255, 255, 0));
        img(Rect(8 * 50 + 3, 5 * 50 + 3, 50 - 5, 50 - 5)).setTo(Scalar(0, 255, 0));
    }

    int key_value = 0;

    while (1)
    {
        namedWindow("原图", WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
        createTrackbar("操作", "原图", &opt_bar, opt_max_number - 1, on_trackbar_share);
        createTrackbar("参数", "原图", &param_bar, 100, on_trackbar_share);
        on_trackbar_share(0,NULL);
        key_value = waitKey(0);
        if (key_value == 'd' || key_value == 'D'|| key_value == 's' || key_value == 'S')
        {
            opt_bar++;
            if (opt_bar >= opt_max_number) opt_bar = 0;
        }
        if (key_value == 'a' || key_value == 'A'|| key_value == 'W' || key_value == 'w')
        {
            opt_bar--;
            if (opt_bar < 0) opt_bar = opt_max_number - 1;
        }
        if (key_value == 27|| key_value == '\r') break;
    }
    destroyAllWindows();
    return 0;
}

