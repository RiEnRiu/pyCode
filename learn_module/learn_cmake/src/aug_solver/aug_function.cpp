// Augmentor.cpp : 定义控制台应用程序的入口点。
//
#include"aug_function.h"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<ctime>
#include<random>


int augmentor_png::getAugRand(int min, int max)
{
    static std::default_random_engine e(time(0));
    std::uniform_int_distribution<int> dist(min, max);
    return dist(e);
}

void augmentor_png::getAugRand(int n, int min, int max, std::vector<int> &rand_output)
{
    if (n <= 0) return;
    static std::default_random_engine e(time(0));
    std::uniform_int_distribution<int> dist(min, max);

    rand_output.resize(n);
    for (int i = 0; i < n; ++i)
    {
        rand_output[i] = dist(e);
    }
    return;
}

double augmentor_png::getAugRand(double min, double max)
{
    static std::default_random_engine e(time(0));
    std::uniform_real_distribution<double> dist(min, max);
    return dist(e);
}

void augmentor_png::getAugRand(int n, double min, double max, std::vector<double> rand_output)
{
    if (n <= 0) return;
    static std::default_random_engine e(time(0));
    std::uniform_real_distribution<double> dist(min, max);

    rand_output.resize(n);
    for (int i = 0; i < n; ++i)
    {
        rand_output[i] = dist(e);
    }
    return;
}


void augmentor_png::imCopy(const cv::Mat &src, std::vector<cv::Mat> &outputs, int n)
{
    outputs.resize(n);
    for (int i = 0; i < n; ++i)
    {
        src.copyTo(outputs[i]);
    }
    return;
}

cv::Mat augmentor_png::imRotate_one(const cv::Mat& img, double angle)
{
    cv::Mat newIm;
    int l = (int)sqrtf(img.cols*img.cols + img.rows*img.rows) + 1;   //对角线长度
    cv::Mat bg(l, l, img.type(), cv::Scalar(0, 0, 0, 0));                  //可旋转任意角度的最小黑色背景
    cv::Mat bg_centre(bg, cv::Rect(l / 2.0 - img.cols / 2.0, l / 2.0 - img.rows / 2.0, img.cols, img.rows)); //原图应该在背景的中间位置  
    img.copyTo(bg_centre);                                         //将原图复制到背景的中心  
    cv::Mat r = cv::getRotationMatrix2D(cv::Point2f(l / 2.0, l / 2.0), angle, 1.0);            //绕中心旋转的变换矩阵
    cv::Mat rot_output;
    warpAffine(bg, rot_output, r, bg.size());              //旋转变换到rot_output
    float c = cosf(angle / 180.0*CV_PI);
    float s = sinf(angle / 180.0*CV_PI);
    int newWidth = img.cols *fabs(c) + img.rows*fabs(s);  //输出宽度
    int newHeight = img.rows*fabs(c) + img.cols *fabs(s); //输出高度
    cv::Mat output_center(rot_output, cv::Rect(l / 2.0 - newWidth / 2.0, l / 2.0 - newHeight / 2.0, newWidth, newHeight));//获取旋转后的输出区域；
    output_center.copyTo(newIm);//复制到输出Mat中
    return newIm;
}

cv::Rect augmentor_png::imCrop_rect(const cv::Size &img_size, double percentage)
{
    if (percentage > 1)percentage = 1.0;
    if (percentage < 0)percentage = 0;

    int maxX = img_size.width*(1.0 - percentage);
    int maxY = img_size.height *(1.0 - percentage);

    if (maxX <= 0) maxX = 0;
    if (maxY <= 0) maxY = 0;
    if (maxX >= img_size.width - 1) maxX = img_size.width - 1;
    if (maxY >= img_size.height - 1) maxY = img_size.height - 1;

    cv::Rect oneRect;

    oneRect.x = getAugRand(0, maxX) ;
    oneRect.y = getAugRand(0, maxY);
    oneRect.width = img_size.width * percentage;
    oneRect.height = img_size.height * percentage;

    if (oneRect.x + oneRect.width > img_size.width)oneRect.width = img_size.width - oneRect.x;
    if (oneRect.y + oneRect.height > img_size.height)oneRect.height = img_size.height - oneRect.y;

    return oneRect;
}

cv::Mat augmentor_png::imAffine_one(const cv::Mat& img, int XorY, double offset_rate)
{
    cv::Mat output;
    cv::Mat r;

    cv::Point2f src[3];
    src[0].x = 0.0f; src[0].y = 0.0f;
    src[1].x = img.cols; src[1].y = 0.0f;
    src[2].x = 0.0f; src[2].y = img.rows;
    switch (XorY)
    {
    case X:
    {
        if (offset_rate < 0)
        {
            int offset_x = -offset_rate * img.cols;
            cv::Point2f dist_x[3];
            dist_x[0].x = src[0].x + offset_x; dist_x[0].y = src[0].y;
            dist_x[1].x = src[1].x + offset_x; dist_x[1].y = src[1].y;
            dist_x[2].x = src[2].x; dist_x[2].y = src[2].y;
            r = cv::getAffineTransform(src, dist_x);
            cv::warpAffine(img, output, r, cv::Size(img.cols + offset_x, img.rows));
        }
        else
        {
            int offset_x = offset_rate * img.cols;
            cv::Point2f dist_x[3];
            dist_x[0].x = src[0].x; dist_x[0].y = src[0].y;
            dist_x[1].x = src[1].x; dist_x[1].y = src[1].y;
            dist_x[2].x = src[2].x + offset_x; dist_x[2].y = src[2].y;
            r = cv::getAffineTransform(src, dist_x);
            cv::warpAffine(img, output, r, cv::Size(img.cols + offset_x, img.rows));
        }
        break;
    }
    case Y:
    {
        if (offset_rate < 0)
        {
            int offset_y = -offset_rate * img.rows;
            cv::Point2f dist_y[3];
            dist_y[0].x = src[0].x; dist_y[0].y = src[0].y + offset_y;
            dist_y[1].x = src[1].x; dist_y[1].y = src[1].y;
            dist_y[2].x = src[2].x; dist_y[2].y = src[2].y + offset_y;
            r = cv::getAffineTransform(src, dist_y);
            cv::warpAffine(img, output, r, cv::Size(img.cols, img.rows + offset_y));
        }
        else
        {
            int offset_y = offset_rate * img.rows;
            cv::Point2f dist_y[3];
            dist_y[0].x = src[0].x; dist_y[0].y = src[0].y;
            dist_y[1].x = src[1].x; dist_y[1].y = src[1].y + offset_y;
            dist_y[2].x = src[2].x; dist_y[2].y = src[2].y;
            r = cv::getAffineTransform(src, dist_y);
            cv::warpAffine(img, output, r, cv::Size(img.cols, img.rows + offset_y));
        }
        break;
    }
    }
    return output;
}

cv::Mat augmentor_png::imAddNoise_gauss_one(const cv::Mat &img,double noisiness)
{
    if (img.channels() == 4)
    {
        cv::Mat output = img.clone();
        for (int i = 0; i < img.rows; ++i)
        {
            cv::Vec4b * one_row = output.ptr<cv::Vec4b>(i);
            for (int j = 0; j < img.cols; ++j)
            {
                int noise[3];
                noise[0] = one_row[j][0] + ((getAugRand(0, 510)) - 255.0) * noisiness;
                noise[1] = one_row[j][1] + ((getAugRand(0, 510)) - 255.0) * noisiness;
                noise[2] = one_row[j][2] + ((getAugRand(0, 510)) - 255.0) * noisiness;
                one_row[j][0] = (noise[0] < 0 ? 0 : (noise[0] > 255 ? 255 : noise[0]));
                one_row[j][1] = (noise[1] < 0 ? 0 : (noise[1] > 255 ? 255 : noise[1]));
                one_row[j][2] = (noise[2] < 0 ? 0 : (noise[2] > 255 ? 255 : noise[2]));
            }
        }
        return output;
    }
    else
    {
        cv::Mat output = img.clone();
        for (int i = 0; i < img.rows; ++i)
        {
            cv::Vec3b * one_row = output.ptr<cv::Vec3b>(i);
            for (int j = 0; j < img.cols; ++j)
            {
                int noise[3];
                noise[0] = one_row[j][0] + ((getAugRand(0, 510)) - 255.0) * noisiness;
                noise[1] = one_row[j][1] + ((getAugRand(0, 510)) - 255.0) * noisiness;
                noise[2] = one_row[j][2] + ((getAugRand(0, 510)) - 255.0) * noisiness;
                one_row[j][0] = (noise[0] < 0 ? 0 : (noise[0] > 255 ? 255 : noise[0]));
                one_row[j][1] = (noise[1] < 0 ? 0 : (noise[1] > 255 ? 255 : noise[1]));
                one_row[j][2] = (noise[2] < 0 ? 0 : (noise[2] > 255 ? 255 : noise[2]));
            }
        }
        return output;
    }

}

cv::Mat augmentor_png::imHue_one(const cv::Mat &img, double H_offset)
{
    if (img.channels() == 4)
    {
        //split into bgr a;
        std::vector<cv::Mat> v_bgra;
        cv::split(img, v_bgra);
        cv::Mat bgr;
        cv::Mat a = v_bgra[3].clone();
        v_bgra.pop_back();
        cv::merge(v_bgra, bgr);

        //HLS;
        cv::Mat HLS_8UC3;
        cv::cvtColor(bgr, HLS_8UC3, CV_BGR2HLS);
        for (int i = 0; i < HLS_8UC3.rows; ++i)
        {
            cv::Vec3b * one_row = HLS_8UC3.ptr< cv::Vec3b>(i);
            for (int j = 0; j < HLS_8UC3.cols; ++j)
            {
                double H_real = one_row[j][0] * 2 + H_offset;
                while (H_real < 0) H_real += 360;
                while (H_real > 360) H_real -= 360;
                one_row[j][0] = H_real / 2;
            }
        }

        //bgr a to bgra
        cv::Mat output_bgr;
        cv::cvtColor(HLS_8UC3, output_bgr, CV_HLS2BGR);
        std::vector<cv::Mat > v_output;
        cv::split(output_bgr, v_output);
        v_output.push_back(a);
        cv::Mat output;
        cv::merge(v_output, output);
        return output;
    }
    else
    {
        const cv::Mat bgr = img;

        //HLS;
        cv::Mat HLS_8UC3;
        cv::cvtColor(bgr, HLS_8UC3, CV_BGR2HLS);
        for (int i = 0; i < HLS_8UC3.rows; ++i)
        {
            cv::Vec3b * one_row = HLS_8UC3.ptr< cv::Vec3b>(i);
            for (int j = 0; j < HLS_8UC3.cols; ++j)
            {
                double H_real = one_row[j][0] * 2 + H_offset;
                while (H_real < 0) H_real += 360;
                while (H_real > 360) H_real -= 360;
                one_row[j][0] = H_real / 2;
            }
        }
        //bgr a to bgra
        cv::Mat output_bgr;
        cv::cvtColor(HLS_8UC3, output_bgr, CV_HLS2BGR);
        cv::Mat output = output_bgr;
        return output;
    }
 
}

cv::Mat augmentor_png::imLightness_one(const cv::Mat &img, double L_offset)
{
    if (img.channels() == 4)
    {
        //split into bgr a;
        std::vector<cv::Mat> v_bgra;
        cv::split(img, v_bgra);
        cv::Mat bgr;
        cv::Mat a = v_bgra[3].clone();
        v_bgra.pop_back();
        cv::merge(v_bgra, bgr);

        //HLS;
        cv::Mat HLS_8UC3;
        cv::cvtColor(bgr, HLS_8UC3, CV_BGR2HLS);
        for (int i = 0; i < HLS_8UC3.rows; ++i)
        {
            cv::Vec3b * one_row = HLS_8UC3.ptr< cv::Vec3b>(i);
            for (int j = 0; j < HLS_8UC3.cols; ++j)
            {
                double L_real = (one_row[j][1] / 255.0 + L_offset);
                if (L_real < 0) L_real = 0;
                if (L_real > 1) L_real = 1;
                one_row[j][1] = L_real * 255;
            }
        }

        //bgr a to bgra
        cv::Mat output_bgr;
        cv::cvtColor(HLS_8UC3, output_bgr, CV_HLS2BGR);
        std::vector<cv::Mat > v_output;
        cv::split(output_bgr, v_output);
        v_output.push_back(a);
        cv::Mat output;
        cv::merge(v_output, output);
        return output;
    }
    else
    {
        const cv::Mat bgr = img;
        //HLS;
        cv::Mat HLS_8UC3;
        cv::cvtColor(bgr, HLS_8UC3, CV_BGR2HLS);
        for (int i = 0; i < HLS_8UC3.rows; ++i)
        {
            cv::Vec3b * one_row = HLS_8UC3.ptr< cv::Vec3b>(i);
            for (int j = 0; j < HLS_8UC3.cols; ++j)
            {
                double L_real = (one_row[j][1] / 255.0 + L_offset);
                if (L_real < 0) L_real = 0;
                if (L_real > 1) L_real = 1;
                one_row[j][1] = L_real * 255;
            }
        }

        //bgr a to bgra
        cv::Mat output_bgr;
        cv::cvtColor(HLS_8UC3, output_bgr, CV_HLS2BGR);
        cv::Mat output = output_bgr;
        return output;
    }
  
}

cv::Mat augmentor_png::imSaturation_one(const cv::Mat &img, double S_offset)
{
    if (img.channels() == 4)
    {
        //split into bgr a;
        std::vector<cv::Mat> v_bgra;
        cv::split(img, v_bgra);
        cv::Mat bgr;
        cv::Mat a = v_bgra[3].clone();
        v_bgra.pop_back();
        cv::merge(v_bgra, bgr);

        //HLS;
        cv::Mat HLS_8UC3;
        cv::cvtColor(bgr, HLS_8UC3, CV_BGR2HLS);
        for (int i = 0; i < HLS_8UC3.rows; ++i)
        {
            cv::Vec3b * one_row = HLS_8UC3.ptr< cv::Vec3b>(i);
            for (int j = 0; j < HLS_8UC3.cols; ++j)
            {
                double S_real = (one_row[j][2] / 255.0 + S_offset);
                if (S_real < 0) S_real = 0;
                if (S_real > 1) S_real = 1;
                one_row[j][2] = S_real * 255;
            }
        }

        //bgr a to bgra
        cv::Mat output_bgr;
        cv::cvtColor(HLS_8UC3, output_bgr, CV_HLS2BGR);
        std::vector<cv::Mat > v_output;
        cv::split(output_bgr, v_output);
        v_output.push_back(a);
        cv::Mat output;
        cv::merge(v_output, output);
        return output;
    }
    else
    {
        const cv::Mat bgr = img;

        //HLS;
        cv::Mat HLS_8UC3;
        cv::cvtColor(bgr, HLS_8UC3, CV_BGR2HLS);
        for (int i = 0; i < HLS_8UC3.rows; ++i)
        {
            cv::Vec3b * one_row = HLS_8UC3.ptr< cv::Vec3b>(i);
            for (int j = 0; j < HLS_8UC3.cols; ++j)
            {
                double S_real = (one_row[j][2] / 255.0 + S_offset);
                if (S_real < 0) S_real = 0;
                if (S_real > 1) S_real = 1;
                one_row[j][2] = S_real * 255;
            }
        }

        //bgr a to bgra
        cv::Mat output_bgr;
        cv::cvtColor(HLS_8UC3, output_bgr, CV_HLS2BGR);
        cv::Mat output = output_bgr;
        return output;
    }
 

}

cv::Mat augmentor_png::imDistort_one(const cv::Mat &input, double rate)
{
    int box_size = 30;
    int rows = input.rows, cols = input.cols;
    int m = (cols - cols / box_size*box_size < 1.0 / 3.0*box_size) ? (cols / box_size + 1) : (cols / box_size + 2);
    int n = (rows - rows / box_size*box_size <1.0 / 3.0*box_size) ? (rows / box_size + 1) : (rows / box_size + 2);
    std::vector<std::vector<cv::Point2f>> src_p(m);
    std::vector<std::vector<cv::Point2f>> dist_p(m);
    for (int i = 0; i < m; ++i) { src_p[i].resize(n);   dist_p[i].resize(n); }
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            src_p[i][j].x = box_size * i;
            src_p[i][j].y = box_size * j;
            if (src_p[i][j].x > cols - 1) { src_p[i][j].x = cols - 1; }
            if (src_p[i][j].y > rows - 1) { src_p[i][j].y = rows - 1; }
            dist_p[i][j].x = box_size * i;
            dist_p[i][j].y = box_size * j;
            if (dist_p[i][j].x > cols - 1) { dist_p[i][j].x = cols - 1; }
            if (dist_p[i][j].y > rows - 1) { dist_p[i][j].y = rows - 1; }
        }
    }

    for (int i = 1; i < m - 1; ++i)
    {
        for (int j = 1; j < n - 1; ++j)
        {
            src_p[i][j].x += (getAugRand(0, 1) * box_size*2.0 / 3.0 - box_size*1.0 / 3.0)*(1 - abs(i - double(m) / 2) / double(m) * 2)*rate;
            src_p[i][j].y += (getAugRand(0, 1) * box_size*2.0 / 3.0 - box_size*1.0 / 3.0)*(1 - abs(j - double(n) / 2) / double(n) * 2)*rate;
        }
    }

    cv::Mat map_x(rows, cols, CV_32FC1);
    cv::Mat map_y(rows, cols, CV_32FC1);
    //printf("%d    %d    %d    %d\n", int(input.rows), int(input.cols),m,n);
    for (int i = 0; i < m - 1; ++i)
    {
        for (int j = 0; j < n - 1; ++j)
        {
            //Down triangle
            {
                cv::Mat src(6, 1, CV_64FC1);
                src.ptr<double>(0)[0] = src_p[i][j].y;
                src.ptr<double>(1)[0] = src_p[i][j].x;
                src.ptr<double>(2)[0] = src_p[i][j + 1].y;
                src.ptr<double>(3)[0] = src_p[i][j + 1].x;
                src.ptr<double>(4)[0] = src_p[i + 1][j + 1].y;
                src.ptr<double>(5)[0] = src_p[i + 1][j + 1].x;

                cv::Mat dist(6, 6, CV_64FC1);
                dist.setTo(0);

                dist.ptr<double>(0)[0] = dist_p[i][j].x; dist.ptr<double>(0)[1] = dist_p[i][j].y; dist.ptr<double>(0)[2] = 1.0;
                dist.ptr<double>(1)[3] = dist_p[i][j].x; dist.ptr<double>(1)[4] = dist_p[i][j].y; dist.ptr<double>(1)[5] = 1.0;
                dist.ptr<double>(2)[0] = dist_p[i][j + 1].x; dist.ptr<double>(2)[1] = dist_p[i][j + 1].y; dist.ptr<double>(2)[2] = 1.0;
                dist.ptr<double>(3)[3] = dist_p[i][j + 1].x; dist.ptr<double>(3)[4] = dist_p[i][j + 1].y; dist.ptr<double>(3)[5] = 1.0;
                dist.ptr<double>(4)[0] = dist_p[i + 1][j + 1].x; dist.ptr<double>(4)[1] = dist_p[i + 1][j + 1].y; dist.ptr<double>(4)[2] = 1.0;
                dist.ptr<double>(5)[3] = dist_p[i + 1][j + 1].x; dist.ptr<double>(5)[4] = dist_p[i + 1][j + 1].y; dist.ptr<double>(5)[5] = 1.0;

                cv::Mat coef = dist.inv() * src;

                double tan_xita = (dist_p[i + 1][j + 1].x - dist_p[i][j].x) / (dist_p[i + 1][j + 1].y - dist_p[i][j].y);
                for (int y = dist_p[i][j].y; y <= dist_p[i + 1][j + 1].y; ++y)
                {
                    float * map_x_line = map_x.ptr<float>(y);
                    float * map_y_line = map_y.ptr<float>(y);
                    for (int x = dist_p[i][j].x; x <= (y - dist_p[i][j].y)*tan_xita + dist_p[i][j].x; ++x)
                    {
                        map_x_line[x] = coef.ptr<double>(3)[0] * x + coef.ptr<double>(4)[0] * y + coef.ptr<double>(5)[0];
                        map_y_line[x] = coef.ptr<double>(0)[0] * x + coef.ptr<double>(1)[0] * y + coef.ptr<double>(2)[0];
                        if (map_x_line[x] < 0) { map_x_line[x] = 0.0f; }
                        if (map_x_line[x] > cols - 1) { map_x_line[x] = cols - 1; }
                        if (map_y_line[x] < 0) { map_y_line[x] = 0.0f; }
                        if (map_y_line[x] > rows - 1) { map_y_line[x] = rows - 1; }
                    }
                }
            }
            //Up triangle
            {
                cv::Mat src(6, 1, CV_64FC1);
                src.ptr<double>(0)[0] = src_p[i][j].y;
                src.ptr<double>(1)[0] = src_p[i][j].x;
                src.ptr<double>(2)[0] = src_p[i + 1][j].y;
                src.ptr<double>(3)[0] = src_p[i + 1][j].x;
                src.ptr<double>(4)[0] = src_p[i + 1][j + 1].y;
                src.ptr<double>(5)[0] = src_p[i + 1][j + 1].x;

                cv::Mat dist(6, 6, CV_64FC1);
                dist.setTo(0);

                dist.ptr<double>(0)[0] = dist_p[i][j].x; dist.ptr<double>(0)[1] = dist_p[i][j].y; dist.ptr<double>(0)[2] = 1.0;
                dist.ptr<double>(1)[3] = dist_p[i][j].x; dist.ptr<double>(1)[4] = dist_p[i][j].y; dist.ptr<double>(1)[5] = 1.0;
                dist.ptr<double>(2)[0] = dist_p[i + 1][j].x; dist.ptr<double>(2)[1] = dist_p[i + 1][j].y; dist.ptr<double>(2)[2] = 1.0;
                dist.ptr<double>(3)[3] = dist_p[i + 1][j].x; dist.ptr<double>(3)[4] = dist_p[i + 1][j].y; dist.ptr<double>(3)[5] = 1.0;
                dist.ptr<double>(4)[0] = dist_p[i + 1][j + 1].x; dist.ptr<double>(4)[1] = dist_p[i + 1][j + 1].y; dist.ptr<double>(4)[2] = 1.0;
                dist.ptr<double>(5)[3] = dist_p[i + 1][j + 1].x; dist.ptr<double>(5)[4] = dist_p[i + 1][j + 1].y; dist.ptr<double>(5)[5] = 1.0;

                cv::Mat coef = dist.inv() * src;

                double tan_xita = (dist_p[i + 1][j + 1].x - dist_p[i][j].x) / (dist_p[i + 1][j + 1].y - dist_p[i][j].y);
                for (int y = dist_p[i][j].y; y <= dist_p[i + 1][j + 1].y; ++y)
                {
                    float * map_x_line = map_x.ptr<float>(y);
                    float * map_y_line = map_y.ptr<float>(y);
                    for (int x = (y - dist_p[i][j].y)*tan_xita + dist_p[i][j].x; x <= dist_p[i + 1][j + 1].x; ++x)
                    {
                        map_x_line[x] = coef.ptr<double>(3)[0] * x + coef.ptr<double>(4)[0] * y + coef.ptr<double>(5)[0];
                        map_y_line[x] = coef.ptr<double>(0)[0] * x + coef.ptr<double>(1)[0] * y + coef.ptr<double>(2)[0];
                        if (map_x_line[x] < 0) { map_x_line[x] = 0.0f; }
                        if (map_x_line[x] > cols - 1) { map_x_line[x] = cols - 1; }
                        if (map_y_line[x] < 0) { map_y_line[x] = 0.0f; }
                        if (map_y_line[x] > rows - 1) { map_y_line[x] = rows - 1; }
                    }
                }
            }
        }
    }
    cv::Mat output;
    cv::remap(input, output, map_x, map_y, CV_INTER_LINEAR);
    return output;
}


cv::Mat augmentor_png::imPerspective_one(const cv::Mat& img, int type, double offset_rate)
{
    cv::Point2f src[4];
    src[0].x = 0.0f; src[0].y = 0.0f;
    src[1].x = img.cols; src[1].y = 0.0f;
    src[2].x = img.cols; src[2].y = img.rows;
    src[3].x = 0.0f; src[3].y = img.rows;

    cv::Point2f dist[4];
    dist[0].x = 0.0f; dist[0].y = 0.0f;
    dist[1].x = img.cols; dist[1].y = 0.0f;
    dist[2].x = img.cols; dist[2].y = img.rows;
    dist[3].x = 0.0f; dist[3].y = img.rows;


    int offset_x = offset_rate * img.cols;
    int offset_y = offset_rate * img.rows;

    switch (type)
    {
    case U:
        dist[0].x += offset_x;
        dist[1].x -= offset_x;
        break;
    case UR:
        dist[1].x -= offset_x; dist[1].y += offset_y;
        break;
    case R:
        dist[1].y += offset_y;
        dist[2].y -= offset_y;
        break;
    case DR:
        dist[2].x -= offset_x; dist[2].y -= offset_y;
        break;
    case D:
        dist[2].x -= offset_x;
        dist[3].x += offset_x;
        break;
    case DL:
        dist[3].x += offset_x; dist[3].y -= offset_y;
        break;
    case L:
        dist[0].y += offset_y;
        dist[3].y -= offset_y;
        break;
    case UL:
        dist[0].x += offset_x; dist[0].y += offset_y;
        break;
    }
    cv::Mat r = cv::getPerspectiveTransform(src, dist);
    cv::Mat output;
    cv::warpPerspective(img, output, r, cv::Size(img.cols, img.rows));
    return output;
}

void augmentor_png::imRotate(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &angles)
{
    aug_obj.resize(angles.size());
    for (int i = 0; i < angles.size(); ++i)
    {
        aug_obj[i] = imRotate_one(obj, angles[i]);
    }
    return;
}

void augmentor_png::imFlip(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj)
{
    aug_obj.resize(1);
    cv::flip(obj, aug_obj[0], getAugRand(0, 1));
    return;
}

void augmentor_png::imFlipUD(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj)
{
    aug_obj.resize(1);
    cv::flip(obj, aug_obj[0], 0);//flipCode==0
    return;
}

void augmentor_png::imFlipLR(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj)
{
    aug_obj.resize(1);
    cv::flip(obj, aug_obj[0], 1);//flipCode>0
    return;
}

void augmentor_png::imCrop(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rate)
{
    aug_obj.resize(rate.size());
    for (int i = 0; i < rate.size(); ++i)
    {
        cv::Rect one_rand_rect = imCrop_rect(cv::Size(obj.cols, obj.rows), rate[i]);
        obj(one_rand_rect).copyTo(aug_obj[i]);
    }
    return;
}

void augmentor_png::imAffine(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    std::vector<int> XorY_type;
    getAugRand(int(rates.size()),1,2,XorY_type);
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        if (XorY_type[i] == 1) aug_obj[i] = imAffine_one(obj, X, rates[i]);
        else if (XorY_type[i] == 2) aug_obj[i] = imAffine_one(obj, Y, rates[i]);
    }
    return;
}

void augmentor_png::imAffineX(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imAffine_one(obj, X, rates[i]);
    }
    return;
}

void augmentor_png::imAffineY(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imAffine_one(obj, Y, rates[i]);
    }
    return;
}

void augmentor_png::imAddNoise_gauss(const cv::Mat &obj,  std::vector<cv::Mat> &aug_obj,  const std::vector<double> &noisiness)
{
    aug_obj.resize(noisiness.size());
    for (int i = 0; i < noisiness.size(); ++i)
    {
        aug_obj[i] = imAddNoise_gauss_one(obj, noisiness[i]);
    }
    return;
}

void augmentor_png::imHue(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &H_offset)
{
    aug_obj.resize(H_offset.size());
    for (int i = 0; i < H_offset.size(); ++i)
    {
        aug_obj[i] = imHue_one(obj, H_offset[i]);
    }
    return;
}

void augmentor_png::imLightness(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &L_offset)
{
    aug_obj.resize(L_offset.size());
    for (int i = 0; i < L_offset.size(); ++i)
    {
        aug_obj[i] = imLightness_one(obj, L_offset[i]);
    }
    return;
}

void augmentor_png::imSaturation(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &S_offset)
{
    aug_obj.resize(S_offset.size());
    for (int i = 0; i < S_offset.size(); ++i)
    {
        aug_obj[i] = imSaturation_one(obj, S_offset[i]);
    }
    return;
}

void augmentor_png::imPerspectiveUL(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imPerspective_one(obj, UL, rates[i]);
    }
    return;
}

void augmentor_png::imPerspectiveU(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imPerspective_one(obj, U, rates[i]);
    }
    return;
}

void augmentor_png::imPerspectiveUR(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imPerspective_one(obj, UR, rates[i]);
    }
    return;
}

void augmentor_png::imPerspectiveL(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imPerspective_one(obj, L, rates[i]);
    }
    return;
}

void augmentor_png::imPerspective(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    std::vector<int> type;
    getAugRand(rates.size(), 1, 8, type);
    for (int i = 0; i < rates.size(); ++i)
    {
        if (type[i] == 1) aug_obj[i] = imPerspective_one(obj, UL, rates[i]);
        else if (type[i] == 2) aug_obj[i] = imPerspective_one(obj, U, rates[i]);
        else if (type[i] == 3) aug_obj[i] = imPerspective_one(obj, UR, rates[i]);
        else if (type[i] == 4) aug_obj[i] = imPerspective_one(obj, L, rates[i]);
        else if (type[i] == 5) aug_obj[i] = imPerspective_one(obj, R, rates[i]);
        else if (type[i] == 6) aug_obj[i] = imPerspective_one(obj, DL, rates[i]);
        else if (type[i] == 7) aug_obj[i] = imPerspective_one(obj, D, rates[i]);
        else if (type[i] == 8) aug_obj[i] = imPerspective_one(obj, DR, rates[i]);
    }
    return;
}

void augmentor_png::imPerspectiveR(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imPerspective_one(obj, R, rates[i]);
    }
    return;
}

void augmentor_png::imPerspectiveDL(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imPerspective_one(obj, DL, rates[i]);
    }
    return;
}

void augmentor_png::imPerspectiveD(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imPerspective_one(obj, D, rates[i]);
    }
    return;
}

void augmentor_png::imPerspectiveDR(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imPerspective_one(obj, DR, rates[i]);
    }
    return;
}

void augmentor_png::imDistort(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, const std::vector<double> &rates)
{
    aug_obj.resize(rates.size());
    for (int i = 0; i < rates.size(); ++i)
    {
        aug_obj[i] = imDistort_one(obj, rates[i]);
    }
    //printf("yes distort %d\n", int(aug_obj.size()));
    return;
}

void augmentor_png::imPyramid(const cv::Mat &obj, std::vector<cv::Mat> &aug_obj, int down_number, int up_number)
{
    if (down_number < 0)down_number = 0;
    if (up_number < 0)up_number = 0;
    int n = down_number + up_number + 1;
    aug_obj.resize(n);
    obj.copyTo(aug_obj[down_number]);
    for (int i = down_number - 1; i >= 0; --i)
    {
        cv::pyrDown(aug_obj[i + 1], aug_obj[i]);
    }
    for (int i = down_number + 1; i < n; ++i)
    {
        cv::pyrUp(aug_obj[i - 1], aug_obj[i]);
    }
    return;
}
