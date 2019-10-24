#include"mask_maker_gui.h"
#include"globalmatting.h"
#include"guidedfilter.h"

#include"stl_expand/stl_expand.hpp"
#include<string>
#include<vector>
#include<fstream>
#include<iostream>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <boost/program_options.hpp>  
namespace po = boost::program_options;

double mask_maker::rate_param = 750.0 * 500.0;
mask_config mask_maker::config_default;

mask_maker::~mask_maker() {}

bool mask_maker::isAngleInRange(double angle, double minRange, double maxRange)
{
    maxRange = maxRange - int(maxRange / 360.0)*360.0;
    if (maxRange < 0) { maxRange += 360.0; }
    minRange = minRange - int(minRange / 360.0)*360.0;
    if (minRange < 0) { minRange += 360.0; }
    angle = angle - int(angle / 360.0)*360.0;
    if (angle < 0) { angle += 360.0; }
    if (minRange <= maxRange) { return (minRange <= angle && angle <= maxRange); }
    else { return (angle >= maxRange || angle <= minRange); }
}

double mask_maker::angleABS(double angle1, double angle2)
{
    double delta = angle1 - angle2;
    delta = delta - int(delta / 360.0)*360.0;
    if (delta < 0) { delta += 360.0; }
    if (delta > 180.0) { return 360.0 - delta; }
    else { return delta; }
}

cv::Rect mask_maker::findMaxObjectBoundingBox(const cv::Mat &bin_mask_8UC1)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(bin_mask_8UC1, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

    std::vector<std::vector<cv::Point> > contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
    }
    int index = -1;
    double area = -1;
    for (int i = 0; i < boundRect.size(); ++i)
    {
        double t_area = boundRect[i].area();
        if (t_area > area)
        {
            index = i;
            area = t_area;
        }
    }
    if (index == -1)
    {
        std::cout << "mask_maker: findMaxObjectBoundingBox: can not find obj." << std::endl;
        return cv::Rect(0, 0, 0, 0);
    }
    if (boundRect[index].x < 0)
    {
        std::cout << "mask_maker: findMaxObjectBoundingBox: Rect.x < 0" << std::endl;
        boundRect[index].x = 0;
    }
    if (boundRect[index].y < 0)
    {
        std::cout << "mask_maker: findMaxObjectBoundingBox: Rect.y < 0" << std::endl;
        boundRect[index].y = 0;
    }
    if (boundRect[index].width + boundRect[index].x > bin_mask_8UC1.cols)
    {
        std::cout << "mask_maker: findMaxObjectBoundingBox: Rect.width < cols" << std::endl;
        boundRect[index].width = bin_mask_8UC1.cols - boundRect[index].x;
    }
    if (boundRect[index].height + boundRect[index].y > bin_mask_8UC1.rows)
    {
        std::cout << "mask_maker: findMaxObjectBoundingBox: Rect.height < rows" << std::endl;
        boundRect[index].height = bin_mask_8UC1.rows - boundRect[index].y;
    }
    return boundRect[index];
}

cv::Rect mask_maker::findMaskBoundingBox(const cv::Mat &bin_mask_8UC1)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(bin_mask_8UC1, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

    if (contours.size() == 0)
    {
        std::cout << "mask_maker: findMaskBoundingBox: can not find obj." << std::endl;
        return cv::Rect(0, 0, 0, 0);
    }


    std::vector<std::vector<cv::Point> > contours_poly(contours.size());
    std::vector<int> x1(contours.size());
    std::vector<int> y1(contours.size());
    std::vector<int> x2(contours.size());
    std::vector<int> y2(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
        cv::Rect one_rect = cv::boundingRect(cv::Mat(contours_poly[i]));
        x1[i] = one_rect.x;
        y1[i] = one_rect.y;
        x2[i] = x1[i]+ one_rect.width;
        y2[i] = y1[i] + one_rect.height;
    }

    cv::Rect output;
    output.x = *(std::min_element(x1.begin(), x1.end()));
    output.y = *(std::min_element(y1.begin(), y1.end()));
    output.width = *(std::max_element(x2.begin(), x2.end())) - output.x;
    output.height = *(std::max_element(y2.begin(), y2.end())) - output.y;

    return output;
}

cv::Mat mask_maker::getTripMap(const cv::Mat &fit_obj_rgb_8uc3,int bg_threshold, int fg_threshold)
{
    cv::Mat output_trip_map;

    //cv::Mat smooth_fit_obj_rgb_8uc3;
    //cv::blur(fit_obj_rgb_8uc3, smooth_fit_obj_rgb_8uc3, cv::Size(11, 11));
    //cv::Mat smooth_fit_obj_rgb_8uc3 = fit_obj_rgb_8uc3.clone();

    //gradient eidge
    int gradient_ker_size = 9;
    cv::Mat gradient;
    //morphologyEx(fit_obj_rgb_8uc3, gradient, CV_MOP_GRADIENT, cv::getStructuringElement(0, cv::Size(gradient_ker_size, gradient_ker_size)));
    //cv::imshow("gradient", gradient);

    /////////////////////////////////////
    //case1
    //cv::Mat pengzhang;
    //morphologyEx(smooth_fit_obj_rgb_8uc3, pengzhang, CV_MOP_DILATE, cv::getStructuringElement(0, cv::Size(gradient_ker_size, gradient_ker_size)));
    //gradient = pengzhang - smooth_fit_obj_rgb_8uc3;
    //cv::imshow("gradient1", gradient);
    /////////////////////////////////////

    /////////////////////////////////////
    //case4
    cv::Mat fushi;
    morphologyEx(fit_obj_rgb_8uc3, fushi, CV_MOP_ERODE, cv::getStructuringElement(0, cv::Size(gradient_ker_size, gradient_ker_size)));
    gradient = fit_obj_rgb_8uc3 - fushi;
    //cv::imshow("gradient4", gradient);
    /////////////////////////////////////

    /////////////////////////////////////
    //case2
    //cv::Mat hls;
    //cv::cvtColor(smooth_fit_obj_rgb_8uc3, hls, CV_BGR2HLS);
    //morphologyEx(hls, gradient, CV_MOP_GRADIENT, cv::getStructuringElement(0, cv::Size(gradient_ker_size, gradient_ker_size)));
    //cv::cvtColor(gradient, gradient, CV_HLS2BGR);
    //cv::imshow("gradient2", gradient);
    /////////////////////////////////////

    /////////////////////////////////////
    //case5
    //cv::Mat grad_x, grad_y;
    //cv::Mat abs_grad_x, abs_grad_y, dst;
    //cv::Sobel(fit_obj_rgb_8uc3, grad_x, CV_32F, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
    //convertScaleAbs(grad_x, abs_grad_x);
    //cv::Sobel(fit_obj_rgb_8uc3, grad_y, CV_32F, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
    //convertScaleAbs(grad_y, abs_grad_y);
    //cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
    //dst.convertTo(gradient, CV_8UC3);
    //cv::imshow("gradient5", gradient);
    /////////////////////////////////////

    /////////////////////////////////////
    //case6
    //cv::Mat hls = fit_obj_rgb_8uc3.clone();

    //for (int i = 0; i < fit_obj_rgb_8uc3.rows; ++i)
    //{
    //    auto iter = hls.ptr<uchar>(i);
    //    for (int j = 0; j < fit_obj_rgb_8uc3.cols*fit_obj_rgb_8uc3.channels(); ++j)
    //    {
    //        iter[j] = iter[j] / int(20) * 20;
    //    }
    //}
    //cv::cvtColor(hls, hls, CV_BGR2HLS);
    //std::vector<cv::Mat> v_hls;
    //cv::split(hls, v_hls);
    //cv::equalizeHist(v_hls[0], v_hls[0]);
    ////cv::equalizeHist(v_hls[2], v_hls[2]);
    //cv::merge(v_hls, hls);
    //cv::cvtColor(hls, gradient, CV_HLS2BGR);
    //morphologyEx(gradient, fushi, CV_MOP_ERODE, cv::getStructuringElement(0, cv::Size(gradient_ker_size, gradient_ker_size)));
    //gradient = gradient - fushi;
    //cv::imshow("gradient6", gradient);
    /////////////////////////////////////

    cv::Mat gray;
    cvtColor(gradient, gray, CV_BGR2GRAY);

    /////////////////////////////////////
    //case3
    //cv::Mat hls;
    //cv::cvtColor(fit_obj_rgb_8uc3, hls, CV_BGR2HLS);
    //std::vector<cv::Mat> v_hls;
    //cv::split(hls, v_hls);
    //morphologyEx(v_hls[0], gray, CV_MOP_GRADIENT, cv::getStructuringElement(0, cv::Size(gradient_ker_size, gradient_ker_size)));
    //cv::imshow("gradient3", gray);
    /////////////////////////////////////

    cv::Mat thre;
    cv::threshold(gray, thre, bg_threshold, 255, CV_THRESH_BINARY);

    cv::Mat t = thre.clone();
    cv::Rect rect = mask_maker::findMaxObjectBoundingBox(t);
    thre.setTo(0);
    t(rect).copyTo(thre(rect));

    //thre1 to dilate thre and then fill the holes in obj
    //thre2 to erode thre
    cv::Mat thre1, thre2;
    thre.copyTo(thre1);
    morphologyEx(thre1, thre1, CV_MOP_DILATE, cv::getStructuringElement(0, cv::Size(5, 5)));
    cv::floodFill(thre1, cv::Point(0, 0), cv::Scalar(127));
    //if (thre1.at<uchar>(0, thre1.cols - 1) != 127) cv::floodFill(thre1, cv::Point(thre1.cols - 1, 0), cv::Scalar(127));//Point and at<> are different;
    //if (thre1.at<uchar>(thre1.rows - 1, 0) != 127) cv::floodFill(thre1, cv::Point(0, thre1.rows - 1), cv::Scalar(127));//Point and at<> are different;
    //if (thre1.at<uchar>(thre1.rows - 1, thre1.cols - 1) != 127) cv::floodFill(thre1, cv::Point(thre1.cols - 1, thre1.rows - 1), cv::Scalar(127));//Point and at<> are different;

    if (thre1.ptr<uchar>(0)[thre1.cols - 1] != 127) cv::floodFill(thre1, cv::Point(thre1.cols - 1, 0), cv::Scalar(127));//Point and at<> are different;
    if (thre1.ptr<uchar>(thre1.rows - 1)[0] != 127) cv::floodFill(thre1, cv::Point(0, thre1.rows - 1), cv::Scalar(127));//Point and at<> are different;
    if (thre1.ptr<uchar>(thre1.rows - 1)[thre1.cols - 1] != 127) cv::floodFill(thre1, cv::Point(thre1.cols - 1, thre1.rows - 1), cv::Scalar(127));//Point and at<> are different;

    cv::threshold(thre1, thre1, 254, -1, CV_THRESH_TOZERO_INV);
    cv::threshold(thre1, thre1, 100, 255, CV_THRESH_BINARY_INV);

    thre1.copyTo(thre2);

    for (int i = 0; i < fg_threshold; ++i)
    {
        cv::morphologyEx(thre2, thre2, CV_MOP_ERODE, cv::getStructuringElement(0, cv::Size(9, 9)));
    }

    output_trip_map.create(thre1.rows, thre1.cols, CV_8UC1);
    double t_count[3] = { 0.0, 0.0 ,0.0 };
    for (int i = 0; i < thre1.rows; ++i)
    {
        uchar * thre1_line = thre1.ptr<uchar>(i);
        uchar * thre2_line = thre2.ptr<uchar>(i);
        uchar * output_trip_map_line = output_trip_map.ptr<uchar>(i);
        for (int j = 0; j < thre1.cols; ++j)
        {
            int t = int(thre1_line[j] + thre2_line[j]) / 2;
            output_trip_map_line[j] = t;
            if (t == 0) { t_count[0] += 1.0; }
            else if (t == 255) { t_count[2] += 1.0; }
            else { t_count[1] += 1.0; }
        }
    }
    bool flag;
    double area = thre1.rows*thre1.cols;
    flag = t_count[0] /0.005  > area;
    flag = flag&&t_count[1] / 0.005 > area;
    flag = flag&&t_count[2] / 0.005 > area;

    //std::cout << t_count[0] << "    " << t_count[1] << "    " << t_count[2] << "    " << area << std::endl;

    if (flag)
    {
        //cv::imshow("output_trip_map", output_trip_map);
        return output_trip_map;
    }
    else
    {
        return cv::Mat();
    }
}

cv::Mat mask_maker::outlineMatting(const cv::Mat &fit_obj_rgb_8uc3,
    int bg_threshold,
    int fg_threshold,
    int edge_smooth_kernal_size,
    int mask_threshold)
{
    cv::Mat trimap = getTripMap(fit_obj_rgb_8uc3, bg_threshold, fg_threshold);

    if (trimap.empty())
    {
        return trimap;
    }
    expansionOfKnownRegions(fit_obj_rgb_8uc3, trimap, 9);
    cv::Mat alpha;
    global_Matting(fit_obj_rgb_8uc3, trimap, cv::Mat(), alpha);

    // filter the result with fast guided filter
    alpha = guidedFilter(fit_obj_rgb_8uc3, alpha, 10, 1e-5);


    ////////////////////////////////////////////////
    //cv::Mat bgra = fit_bgr_obj.clone();
    //for (int i = 0; i < bgra.rows; ++i)
    //{
    //    for (int j = 0; j < bgra.cols; ++j)
    //    {
    //        double _rate = double(alpha.at<uchar>(i, j)) / 255;
    //        bgra.at<cv::Vec3b>(i, j)[0] *= _rate;
    //        bgra.at<cv::Vec3b>(i, j)[1] *= _rate;
    //        bgra.at<cv::Vec3b>(i, j)[2] *= _rate;
    //    }
    //}
    //imshow("bgra", bgra);
    ////////////////////////////////////////////////

    //for (int x = 0; x < trimap.cols; ++x)
    //{
    //    for (int y = 0; y < trimap.rows; ++y)
    //    {
    //        if (trimap.at<uchar>(y, x) == 0)
    //            alpha.at<uchar>(y, x) = 0;
    //        else if (trimap.at<uchar>(y, x) == 255)
    //            alpha.at<uchar>(y, x) = 255;
    //    }
    //}

    cv::Mat mask;
    threshold(alpha, mask, 127, 255, CV_THRESH_BINARY);
    cv::blur(mask, mask, cv::Size(edge_smooth_kernal_size, edge_smooth_kernal_size));
    threshold(mask, mask, 127, 255, CV_THRESH_BINARY);

    //cv::Mat black_bg = input.clone();
    //black_bg.setTo(cv::Scalar(0, 0, 255));
    //input.copyTo(black_bg, mask);
    //cv::imshow("obj", input);
    //cv::imshow("mask", mask);
    //cv::imshow("obj_mask", black_bg);
    //cv::waitKey(0);

    //temp_mat.resize(2);
    //trimap.copyTo(temp_mat[0]);
    //cv::resize(mask, temp_mat[1], org_size);

    //
    cv::morphologyEx(mask, mask, CV_MOP_OPEN, cv::getStructuringElement(0, cv::Size(5, 5)));
    cv::morphologyEx(mask, mask, CV_MOP_CLOSE, cv::getStructuringElement(0, cv::Size(5, 5)));
    for (int i = 0; i < 50 - mask_threshold; ++i)
    {
        cv::morphologyEx(mask, mask, CV_MOP_ERODE, cv::getStructuringElement(0, cv::Size(3, 3)));
    }
    for (int i = 0; i < mask_threshold - 50; ++i)
    {
        cv::morphologyEx(mask, mask, CV_MOP_DILATE, cv::getStructuringElement(0, cv::Size(3, 3)));
    }
    cv::threshold(mask, mask, 50, 255, CV_THRESH_BINARY);

    //treat it because of holes in mask
    cv::Mat tripmap_255_mask(trimap.size(),CV_8UC1);
    tripmap_255_mask.setTo(0);
    cv::Mat tripmap_0_mask(trimap.size(), CV_8UC1);
    tripmap_0_mask.setTo(0);
    cv::threshold(trimap, tripmap_255_mask, 200, 255, CV_THRESH_BINARY);
    cv::threshold(trimap, tripmap_0_mask, 100, 255, CV_THRESH_BINARY_INV);
    trimap.copyTo(mask, tripmap_255_mask);
    trimap.copyTo(mask, tripmap_0_mask);


    return mask;

}

void mask_maker::cleanBgHue(const cv::Mat &obj_rgb_8uc3, const std::vector<std::vector<cv::Point>> &invariant, const cv::Mat &outline, std::vector<cv::Mat> &v_bgr_output)
{
    cv::Mat smooth_obj_rbg_8uc3;
    cv::blur(obj_rgb_8uc3, smooth_obj_rbg_8uc3, cv::Size(9, 9));
    cv::Mat smooth_obj_hls_rbg_8uc3;
    cvtColor(smooth_obj_rbg_8uc3, smooth_obj_hls_rbg_8uc3, CV_BGR2HLS);

    //bg_H
    int bg_sample_point = 0;
    double BG_H = 0;//double L = 0;//double S = 0;
    cv::Mat_<cv::Vec3b>::iterator iter_smooth_obj_hls_rbg_8uc3 = smooth_obj_hls_rbg_8uc3.begin<cv::Vec3b>();
    const cv::Mat_<cv::Vec3b>::const_iterator iter_end_smooth_obj_hls_rbg_8uc3 = smooth_obj_hls_rbg_8uc3.end<cv::Vec3b>();
    cv::Mat_ <uchar>::const_iterator iter_outline = outline.begin<uchar>();
    const cv::Mat_<uchar>::const_iterator iter_end_outline = outline.end<uchar>();
    for (; 
        iter_smooth_obj_hls_rbg_8uc3 != iter_end_smooth_obj_hls_rbg_8uc3;
        ++iter_smooth_obj_hls_rbg_8uc3,++iter_outline)
    {
        if ((*iter_outline) == 0 && rand() < RAND_MAX / 100.0)
        {
            ++bg_sample_point;
            BG_H += (*iter_smooth_obj_hls_rbg_8uc3)[0];//L += (*iter_smooth_obj_hls_rbg_8uc3)[1];//S += (*iter_smooth_obj_hls_rbg_8uc3)[2];
        }
    }
    BG_H /= bg_sample_point;// L /= bg_sample_point;// S /= bg_sample_point;
    BG_H *= 2;//because opencv h range is [0,180]

    cv::Mat obj_hls_8uc3;
    cvtColor(obj_rgb_8uc3, obj_hls_8uc3, CV_BGR2HLS);
    cv::Mat_<cv::Vec3b>::iterator iter_obj_hls_8uc3 = obj_hls_8uc3.begin<cv::Vec3b>();
    const cv::Mat_<cv::Vec3b>::const_iterator iter_end_obj_hls_8uc3 = obj_hls_8uc3.end<cv::Vec3b>();
    iter_smooth_obj_hls_rbg_8uc3 = smooth_obj_hls_rbg_8uc3.begin<cv::Vec3b>();
    iter_outline = outline.begin<uchar>();

    for (;
        iter_obj_hls_8uc3 != iter_end_obj_hls_8uc3;
        ++iter_smooth_obj_hls_rbg_8uc3,++iter_obj_hls_8uc3,++iter_outline)
    {
        if ((*iter_outline) != 0)
        {
            int one_H = (*iter_smooth_obj_hls_rbg_8uc3)[0] * 2;//because opencv h range is [0,180]
            if (isAngleInRange(one_H, BG_H - 10, BG_H + 10))
            {
                (*iter_obj_hls_8uc3)[2] = 0;
            }
            else if (isAngleInRange(one_H, BG_H - 30, BG_H + 30))
            {
                (*iter_obj_hls_8uc3)[2] *= (angleABS(BG_H, one_H) - 10) / 20.0;
            }
        }
    }

    cv::Mat output_obj_rgb_8uc3;
    cv::cvtColor(obj_hls_8uc3, output_obj_rgb_8uc3, CV_HLS2BGR);
    cv::Mat obj_rgb_8uc3_hue_mask = p8AreaMask(obj_rgb_8uc3.size(), invariant);
    obj_rgb_8uc3.copyTo(output_obj_rgb_8uc3, obj_rgb_8uc3_hue_mask);

    cv::split(output_obj_rgb_8uc3, v_bgr_output);

    return;
}

cv::Mat mask_maker::alphaMatting(const cv::Mat &fit_obj_rgb_8uc3, const std::vector<std::vector<cv::Point>> &fit_transparent, const cv::Mat &fit_outline)
{
    if (fit_transparent.size() == 0) return fit_outline.clone();

    //gray_mask
    cv::Mat gray_mask = p8AreaMask(fit_obj_rgb_8uc3.size(), fit_transparent);
    cv::Mat inv_fit_outline;
    cv::threshold(fit_outline, inv_fit_outline, 127, 255, CV_THRESH_BINARY_INV);
    fit_outline.copyTo(gray_mask, inv_fit_outline);

    //trip map
    cv::Mat trip_map = fit_outline.clone();
    trip_map.setTo(127, gray_mask);

    //check if trip is vaild
    cv::Rect mask_rect = findMaxObjectBoundingBox(fit_outline);
    const cv::Mat trip_map_roi = trip_map(mask_rect);
    bool zero255 = true;
    const uchar *trip_map_roi_data = trip_map_roi.data;
    for (int i = 0, count255 = 0; i < trip_map_roi.size().area() && zero255; ++i)
    {
        if (trip_map_roi_data[i] == 255)
        {
            count255++;
            if (count255 > 25) { zero255 = false; }
        }
    }

    cv::Mat output;
    cv::Mat trip_map_temp = trip_map.clone();
    if (!zero255)
    {
        expansionOfKnownRegions(fit_obj_rgb_8uc3, trip_map, 9);
        cv::Mat alpha;
        global_Matting(fit_obj_rgb_8uc3, trip_map, cv::Mat(), alpha);
        // filter the result with fast guided filter
        alpha = guidedFilter(fit_obj_rgb_8uc3, alpha, 10, 1e-5);
        output = fit_outline.clone();
        alpha.copyTo(output, gray_mask);
    }
    else
    {
        bool col_div = mask_rect.width > mask_rect.height ? true : false;
        cv::Rect rect1;
        cv::Rect rect2;
        if (col_div)
        {
            rect1.x = mask_rect.x; 
            rect1.y = mask_rect.y; 
            rect1.width = mask_rect.width / 2; 
            rect1.height = mask_rect.height;

            rect2.x = mask_rect.x + mask_rect.width / 2; 
            rect2.y = mask_rect.y; 
            rect2.width = mask_rect.width - mask_rect.width / 2; 
            rect2.height = mask_rect.height;
        }
        else
        {
            rect1.x = mask_rect.x;
            rect1.y = mask_rect.y;
            rect1.width = mask_rect.width; 
            rect1.height = mask_rect.height / 2;

            rect2.x = mask_rect.x;
            rect2.y = mask_rect.height / 2+ mask_rect.y;
            rect2.width = mask_rect.width; 
            rect2.height = mask_rect.height - mask_rect.height / 2;
        }
        cv::Mat trip_map_1 = fit_outline.clone();
        cv::Mat trip_map_2 = fit_outline.clone();
        trip_map_1(rect1).setTo(127, gray_mask(rect1));
        trip_map_2(rect2).setTo(127, gray_mask(rect2));

        output = fit_outline.clone();

        cv::Mat  alpha1;
        expansionOfKnownRegions(fit_obj_rgb_8uc3, trip_map_1, 9);
        global_Matting(fit_obj_rgb_8uc3, trip_map_1, cv::Mat(), alpha1);
        alpha1 = guidedFilter(fit_obj_rgb_8uc3, alpha1, 10, 1e-5);
        alpha1(rect1).copyTo(output(rect1), gray_mask(rect1));

        cv::Mat  alpha2;
        expansionOfKnownRegions(fit_obj_rgb_8uc3, trip_map_2, 9);
        global_Matting(fit_obj_rgb_8uc3, trip_map_2, cv::Mat(), alpha2);
        alpha2 = guidedFilter(fit_obj_rgb_8uc3, alpha2, 10, 1e-5);
        alpha2(rect2).copyTo(output(rect2), gray_mask(rect2));
    }

    //adjust the lower limit
    double a_avg = 0;
    size_t number = 0;
    uchar * p_trip_map_temp = trip_map_temp.data;
    uchar * p_output = output.data;
    for (int i = 0; i < trip_map_temp.size().area(); ++i)
    {
        if (p_trip_map_temp[i] == uchar(127))
        {
            number++;
            a_avg += p_output[i];
            //if (p_output[i] > max_output1) { max_output1 = p_output[i]; }
            //if (p_output[i] < min_output1) { min_output1 = p_output[i]; }
        }
    }
    a_avg /= number;
    //double max_output2 = 0;
    //double min_output2 = 255;
    double bias = 180 - a_avg;
    if (bias > 0)
    {
        for (int i = 0; i < trip_map_temp.size().area(); ++i)
        {
            if (p_trip_map_temp[i] == uchar(127))
            {
                double t = p_output[i] + bias;
                if (t > 255) { p_output[i] = 255; }
                else { p_output[i] = t; }
                //if (p_output[i] > max_output2) { max_output2 = p_output[i]; }
                //if (p_output[i] < min_output2) { min_output2 = p_output[i]; }
            }
        }
    }

    return output;

}

cv::Mat mask_maker::Matting(const std::string &obj_path, const mask_config &config)
{
    cv::Mat obj_rgb_8uc3 = cv::imread(obj_path);
    if (obj_rgb_8uc3.empty()) { return obj_rgb_8uc3; }

    int bg_threshold = config.param.at(CONFIG_KEY_bg_threshold);
    if (0 > bg_threshold || bg_threshold > 255) 
    {
        bg_threshold = config_default.param.at(CONFIG_KEY_bg_threshold);
    }//[0,255]
    int fg_threshold = config.param.at(CONFIG_KEY_fg_threshold);
    if (0 > fg_threshold || fg_threshold > 30) 
    { 
        fg_threshold = config_default.param.at(CONFIG_KEY_fg_threshold);
    }//[0,30]
    int edge_smooth_kernal_size = config.param.at(CONFIG_KEY_edge_smooth_kernal_size);
    if (3 > edge_smooth_kernal_size || edge_smooth_kernal_size > 30) 
    {
        edge_smooth_kernal_size = config_default.param.at(CONFIG_KEY_edge_smooth_kernal_size);
    }//[3,30]
    int mask_threshold = config.param.at(CONFIG_KEY_mask_threshold);
    if (0 > mask_threshold || mask_threshold > 100) 
    {
        mask_threshold = config_default.param.at(CONFIG_KEY_mask_threshold);
    }//[0,100] 

    //fit
    cv::Mat fit_obj_rgb_8uc3;
    double rate = sqrt(rate_param / obj_rgb_8uc3.cols / obj_rgb_8uc3.rows);
    cv::Size fit_size(obj_rgb_8uc3.cols*rate, obj_rgb_8uc3.rows*rate);
    cv::resize(obj_rgb_8uc3, fit_obj_rgb_8uc3, fit_size);
    std::vector<std::vector<cv::Point>> fit_transparent(config.___transparent.size());
    for(int i=0;i<config.___transparent.size();++i){ p8Resize(config.___transparent[i], fit_transparent[i], rate, rate); }
    std::vector<std::vector<cv::Point>> fit_erase_area(config.___erase_area.size());
    for (int i = 0; i<config.___erase_area.size(); ++i) { p8Resize(config.___erase_area[i], fit_erase_area[i], rate, rate); }

    //outline
    cv::Mat fit_outline = outlineMatting(fit_obj_rgb_8uc3, bg_threshold, fg_threshold, edge_smooth_kernal_size, mask_threshold);
    if (fit_outline.empty())
    {
        cv::Mat bgra(obj_rgb_8uc3.rows, obj_rgb_8uc3.cols, CV_8UC4);
        bgra.setTo(cv::Scalar(0, 0, 0, 0));
        return bgra;
    }
    for (int i = 0; i < fit_erase_area.size(); ++i)
    {
        cv::fillConvexPoly(fit_outline, fit_erase_area[i], cv::Scalar(0));
    }

    cv::Size org_size(obj_rgb_8uc3.cols, obj_rgb_8uc3.rows);
    cv::Mat outline;
    cv::resize(fit_outline, outline, org_size);
    cv::blur(outline, outline, cv::Size(edge_smooth_kernal_size, edge_smooth_kernal_size));

    std::vector<cv::Mat> v_bgra;
    cleanBgHue(obj_rgb_8uc3, config.___invariant, outline, v_bgra);
   
    cv::Mat fit_a = alphaMatting(fit_obj_rgb_8uc3, fit_transparent, fit_outline);

    cv::Mat a = fit_outline.clone();
    cv::resize(fit_a, a, org_size);

    cv::Mat output_bgra;
    v_bgra.push_back(a);
    cv::merge(v_bgra, output_bgra);

    return output_bgra;
}

cv::Mat mask_maker::Matting(const mask_config &config)
{
    return Matting(config.obj_path, config);
}

cv::Mat mask_maker::getInvariantPattern(const cv::Mat &input)
{
    static cv::Mat pattern;
    if (pattern.cols == input.cols && pattern.rows == input.rows)
    {
        return pattern;
    }
    else
    {
        int rows = input.rows;
        int cols = input.cols;
        pattern.create(rows, cols, CV_8UC1);
        const int part_size = 10;
        int rows_parts = rows / part_size;
        int cols_parts = cols / part_size;
        for (int i = 0; i < rows; i++)
        {
            uchar* dataout = pattern.ptr<uchar>(i);
            for (int j = 0; j < cols; j++)
            {
                int ii = (i / part_size) % 2;
                int jj = (j / part_size) % 2;
                if ((ii == 0 && jj == 0)||(ii == 1 && jj == 1))
                {
                    dataout[j] = 255;
                }
                else
                {
                    dataout[j] = 255-64;
                }
            }
        }
        return pattern;
    }
}

bool mask_maker::isIntersect(const cv::Rect &region1, const cv::Rect &region2)
{
    int x11 = region1.x;
    int y11 = region1.y;
    int x12 = region1.x + region1.width;
    int y12 = region1.y + region1.height;
    int x21 = region2.x;
    int y21 = region2.y;
    int x22 = region2.x + region2.width;
    int y22 = region2.y + region2.height;

    if (y22 < y11) { return false; }//r2在r1上边
    if (y21 > y12) { return false; }//r2在r1下边
    if (x22 < x11) { return false; }//r2在r1左边
    if (x21 > x12) { return false; }//r2在r1右边

    return true;
}

bool mask_maker::preview_mask(mask_config &inputOutput_config, double bg_brightness,cv::Mat &view_mat)
{
    cv::Mat obj_8uc4 = make_mask(inputOutput_config);
    if (obj_8uc4.empty()) { return false; }

    cv::Mat &fit_obj_8uc4 = view_mat;
    double rate = sqrt(rate_param / obj_8uc4.cols / obj_8uc4.rows);
    cv::Size fit_size(obj_8uc4.cols*rate, obj_8uc4.rows*rate);
    cv::resize(obj_8uc4, fit_obj_8uc4, fit_size);

    //blue
    for (int i = 0; i < inputOutput_config.___transparent.size(); ++i)
    {
        std::vector<cv::Point> temp;
        p8Resize(inputOutput_config.___transparent[i], temp, rate, rate);
        cv::polylines(fit_obj_8uc4, temp, true, cv::Scalar(255, 0, 0, 255));
    }

    //green
    for (int i = 0; i < inputOutput_config.___invariant.size(); ++i)
    {
        std::vector<cv::Point> temp;
        p8Resize(inputOutput_config.___invariant[i], temp, rate, rate);
        cv::polylines(fit_obj_8uc4, temp, true, cv::Scalar(0, 255, 0, 255));
    }

    //red
    for (int i = 0; i < inputOutput_config.___erase_area.size(); ++i)
    {
        std::vector<cv::Point> temp;
        p8Resize(inputOutput_config.___erase_area[i], temp, rate, rate);
        cv::polylines(fit_obj_8uc4, temp, true, cv::Scalar(0, 0, 255, 255));
    }

    //effect
    imshow_png("抠图参数配置器", fit_obj_8uc4, bg_brightness);
    cv::Mat src = cv::imread(inputOutput_config.obj_path);
    cv::Mat fit_src;
    cv::resize(src, fit_src, fit_size);

    //source image
    std::string label;
    std::string _temp_part;
    std::expand::split2(inputOutput_config.obj_path, "\\/", label, _temp_part, -1);
    std::expand::split2(label, "\\/", _temp_part, label, 1);
    cv::putText(fit_src, label, cv::Point(30, 30), 1, 1, cv::Scalar(0, 0, 255));

    cv::imshow("原始图像", fit_src);

    return true;
}


//bool mask_maker::preview_mask(mask_config &inputOutput_config,double bg_brightness)
//{
//    return preview_mask(inputOutput_config, bg_brightness, cv::Mat());
//}

void mask_maker::split_8UC4(const cv::Mat &bgra, cv::Mat &bgr, cv::Mat &a)
{
    if (bgra.isContinuous())
    {
        bgr.create(bgra.rows, bgra.cols, CV_8UC3);
        a.create(bgra.rows, bgra.cols, CV_8UC1);
        const uchar *p_bgra = bgra.ptr<uchar>();
        uchar *p_bgr = bgr.ptr<uchar>();
        uchar *p_a = a.ptr<uchar>();
        int i_p_bgra = 0, i_p_bgr = 0, i_p_a = 0;
        int n = bgra.rows * bgra.cols;
        for (int i = 0; i < n; ++i)
        {
            p_bgr[i_p_bgr] = p_bgra[i_p_bgra]; ++i_p_bgr; ++i_p_bgra;
            p_bgr[i_p_bgr] = p_bgra[i_p_bgra]; ++i_p_bgr; ++i_p_bgra;
            p_bgr[i_p_bgr] = p_bgra[i_p_bgra]; ++i_p_bgr; ++i_p_bgra;
            p_a[i_p_a] = p_bgra[i_p_bgra]; ++i_p_a; ++i_p_bgra;
        }
    }
    else
    {
        std::vector<cv::Mat> v_bgra;
        cv::split(bgra, v_bgra);
        a = v_bgra[3].clone();
        v_bgra.pop_back();
        cv::merge(v_bgra, bgr);
    }
    return;
}

void mask_maker::merge(const cv::Mat &bgr, const cv::Mat &a, cv::Mat &bgra)
{
    std::vector<cv::Mat> v_bgra;
    cv::split(bgr, v_bgra);
    v_bgra.push_back(a);
    cv::merge(v_bgra, bgra);
    return;
}

cv::Mat mask_maker::png2binMask(const cv::Mat &bgra)
{
    cv::Mat bgr, a, output;
    split_8UC4(bgra, bgr, a);
    cv::threshold(a, output, 100, 255, CV_THRESH_BINARY);
    return output;
}

cv::Mat mask_maker::compactPNG(const cv::Mat &bgra)
{
    cv::Mat mask = png2binMask(bgra);
    cv::Rect rect = findMaskBoundingBox(mask);
    if (rect.area() == 0) { return cv::Mat(); }
    return bgra(rect).clone();
}

void mask_maker::on_Trackbar_config_value_0(int bar_value, void *void_ptr_pack)
{
    void **void_ptr = (void **)void_ptr_pack;
    mask_maker * p_mask_maker = (mask_maker *)void_ptr[0];
    mask_config * p_config = (mask_config *)void_ptr[1];
    cv::Mat *p_effect_mat = (cv::Mat *)void_ptr[3];
    if (bar_value > 50)
    {
        bar_value = 204.0 / 49.0*(bar_value - 100) + 255;
        if (bar_value > 255) { bar_value = 255; }
        else if (bar_value < 0) { bar_value = 0; }
    }
    int *p_bar_4_value = (int *)void_ptr[2];
    p_config->param[CONFIG_KEY_bg_threshold] = bar_value;
    p_mask_maker->preview_mask(*p_config, *p_bar_4_value / 100.0, *p_effect_mat);
    return;
}

void mask_maker::on_Trackbar_config_value_1(int bar_value, void *void_ptr_pack)
{
    void **void_ptr = (void **)void_ptr_pack;
    mask_maker * p_mask_maker = (mask_maker *)void_ptr[0];
    mask_config * p_config = (mask_config *)void_ptr[1];
    cv::Mat *p_effect_mat = (cv::Mat *)void_ptr[3];
    int *p_bar_4_value = (int *)void_ptr[2];
    p_config->param[CONFIG_KEY_fg_threshold] = bar_value;
    p_mask_maker->preview_mask(*p_config, *p_bar_4_value / 100.0, *p_effect_mat);
    return;
}

void mask_maker::on_Trackbar_config_value_2(int bar_value, void *void_ptr_pack)
{
    void **void_ptr = (void **)void_ptr_pack;
    mask_maker * p_mask_maker = (mask_maker *)void_ptr[0];
    mask_config * p_config = (mask_config *)void_ptr[1];
    int *p_bar_4_value = (int *)void_ptr[2];
    cv::Mat *p_effect_mat = (cv::Mat *)void_ptr[3];
    p_config->param[CONFIG_KEY_edge_smooth_kernal_size] = bar_value;
    p_mask_maker->preview_mask(*p_config, *p_bar_4_value / 100.0, *p_effect_mat);
    return;
}

void mask_maker::on_Trackbar_config_value_3(int bar_value, void *void_ptr_pack)
{
    void **void_ptr = (void **)void_ptr_pack;
    mask_maker * p_mask_maker = (mask_maker *)void_ptr[0];
    mask_config * p_config = (mask_config *)void_ptr[1];
    int *p_bar_4_value = (int *)void_ptr[2];
    cv::Mat *p_effect_mat = (cv::Mat *)void_ptr[3];
    p_config->param[CONFIG_KEY_mask_threshold] = bar_value;
    p_mask_maker->preview_mask(*p_config, *p_bar_4_value / 100.0, *p_effect_mat);
    return;
}

void mask_maker::on_Trackbar_config_value_4(int bar_value, void *void_ptr_pack)
{
    void **void_ptr = (void **)void_ptr_pack;
    mask_maker * p_mask_maker = (mask_maker *)void_ptr[0];
    mask_config * p_config = (mask_config *)void_ptr[1];
    cv::Mat *p_effect_mat = (cv::Mat *)void_ptr[3];
    p_mask_maker->preview_mask(*p_config, bar_value / 100.0, *p_effect_mat);
    return;
}

void mask_maker::on_Mouse(int event, int x, int y, int flags, void *void_ptr_pack)
{
    void **void_ptr = (void **)void_ptr_pack;
    mask_maker * p_mask_maker = (mask_maker *)void_ptr[0];
    mask_config * p_config = (mask_config *)void_ptr[1];
    int *tb_4_value = (int  *)void_ptr[2];
    int *tb_5_value = (int  *)void_ptr[3];
    cv::Mat *p_effect_mat = (cv::Mat *)void_ptr[4];
    bool *p_is_shift_obj = (bool *)void_ptr[5];

    static std::vector<cv::Point> l_type0_p(8); 
    static std::vector<cv::Point> m_type0_p(8);
    static std::vector<cv::Point> r_type0_p(8);

    static int i_l_type0_p = 0;
    static int i_m_type0_p = 0;
    static int i_r_type0_p = 0;

    if (*p_is_shift_obj)
    {
        //std::cout << "清零" << std::endl;
        i_l_type0_p = 0; i_m_type0_p = 0; i_r_type0_p = 0;
        *p_is_shift_obj = false;
    }

    static std::vector<cv::Point> l_type1_point(2);
    static std::vector<cv::Point> r_type1_point(2);
    static std::vector<cv::Point> m_type1_point(2);

    static bool is_l_down_type1 = false;
    static bool is_m_down_type1 = false;
    static bool is_r_down_type1 = false;

    //draw once
    if (event == CV_EVENT_MOUSEMOVE)
    {
        cv::Mat temp = p_effect_mat->clone();
        cv::line(temp, cv::Point(x,0), cv::Point(x, p_effect_mat->rows - 1), cv::Scalar(0, 255, 255, 255));
        cv::line(temp, cv::Point(0, y), cv::Point(p_effect_mat->cols - 1, y), cv::Scalar(0, 255, 255, 255));
        
        if (is_l_down_type1)
        {
            cv::rectangle(temp, cv::Rect(l_type1_point[0], cv::Point(x, y)), cv::Scalar(255, 0, 0, 255));
        }
        if (is_m_down_type1)
        {
            cv::rectangle(temp, cv::Rect(m_type1_point[0], cv::Point(x, y)), cv::Scalar(0, 255, 0, 255));
        }
        if (is_r_down_type1)
        {
            cv::rectangle(temp, cv::Rect(r_type1_point[0], cv::Point(x, y)), cv::Scalar(0, 0, 255, 255));
        }
        imshow_png("抠图参数配置器", temp, *tb_5_value / 100.0);
    }
    else
    {
        switch (*tb_4_value)
        {
        case 0:
        {
            if (event == CV_EVENT_LBUTTONDOWN)
            {
                l_type0_p[i_l_type0_p].x = x;
                l_type0_p[i_l_type0_p].y = y;
                cv::circle(*p_effect_mat, l_type0_p[i_l_type0_p], 1, cv::Scalar(0, 255, 255, 255), 3);

                if (i_l_type0_p != 0 && i_l_type0_p != 7)
                {
                    cv::line(*p_effect_mat, l_type0_p[i_l_type0_p], l_type0_p[i_l_type0_p-1], cv::Scalar(255, 0, 0, 255));
                    imshow_png("抠图参数配置器", *p_effect_mat, *tb_5_value / 100.0);
                }
                i_l_type0_p = i_l_type0_p + 1;
            }
            else  if(event == CV_EVENT_LBUTTONDBLCLK)
            {
                //std::cout << i_l_type0_p << std::endl;
                if (i_l_type0_p < 2) { i_l_type0_p = -1; }
                else
                {
                    for (int i = i_l_type0_p; i < 8; ++i)
                    {
                        l_type0_p[i].x = x;
                        l_type0_p[i].y = y;
                    }
                    i_l_type0_p = 8;
                }
            }
            else if (event == CV_EVENT_MBUTTONDOWN)
            {
                m_type0_p[i_m_type0_p].x = x;
                m_type0_p[i_m_type0_p].y = y;
                cv::circle(*p_effect_mat, m_type0_p[i_m_type0_p], 1, cv::Scalar(0, 255, 255, 255), 3);
                if (i_m_type0_p != 0 && i_m_type0_p != 7)
                {
                    cv::line(*p_effect_mat, m_type0_p[i_m_type0_p], m_type0_p[i_m_type0_p - 1], cv::Scalar(0, 255, 0, 255));
                    imshow_png("抠图参数配置器", *p_effect_mat, *tb_5_value / 100.0);
                }
                i_m_type0_p = i_m_type0_p + 1;
            }
            else  if (event == CV_EVENT_MBUTTONDBLCLK)
            {
                if (i_m_type0_p < 2) { i_m_type0_p = -1; }
                else
                {
                    for (int i = i_m_type0_p; i < 8; ++i)
                    {
                        m_type0_p[i].x = x;
                        m_type0_p[i].y = y;
                    }
                    i_m_type0_p = 8;
                }
            }
            else if (event == CV_EVENT_RBUTTONDOWN)
            {
                r_type0_p[i_r_type0_p].x = x;
                r_type0_p[i_r_type0_p].y = y;
                cv::circle(*p_effect_mat, r_type0_p[i_r_type0_p], 1, cv::Scalar(0, 255, 255, 255), 3);
                if (i_r_type0_p != 0 && i_r_type0_p != 7)
                {
                    cv::line(*p_effect_mat, r_type0_p[i_r_type0_p], r_type0_p[i_r_type0_p - 1], cv::Scalar(0, 0, 255, 255));
                    imshow_png("抠图参数配置器", *p_effect_mat, *tb_5_value / 100.0);
                }
                i_r_type0_p = i_r_type0_p + 1;
            }
            else  if (event == CV_EVENT_RBUTTONDBLCLK)
            {
                if (i_r_type0_p < 2) { i_r_type0_p = -1; }
                else
                {
                    for (int i = i_r_type0_p; i < 8; ++i)
                    {
                        r_type0_p[i].x = x;
                        r_type0_p[i].y = y;
                    }
                    i_r_type0_p = 8;
                }
            }

            bool is_preview = false;
            if (i_l_type0_p == -1) { i_l_type0_p = 0; is_preview = true; }
            if (i_m_type0_p == -1) { i_m_type0_p = 0; is_preview = true; }
            if (i_r_type0_p == -1) { i_r_type0_p = 0; is_preview = true; }
            if (i_l_type0_p == 8 || i_m_type0_p == 8 || i_r_type0_p == 8)
            {
                is_preview = true;
                cv::Mat org_img = cv::imread(p_config->obj_path, CV_LOAD_IMAGE_UNCHANGED);
                double inv_rate = sqrt(double(org_img.cols) * org_img.rows / rate_param);
                std::vector<cv::Point> t_p8;
                if (i_l_type0_p == 8)
                {
                    i_l_type0_p = 0; 
                    p_mask_maker->p8Resize(l_type0_p, t_p8, inv_rate, inv_rate);
                    p_config->___transparent.push_back(t_p8);
                }
                if (i_m_type0_p == 8)
                {
                    i_m_type0_p = 0;
                    p_mask_maker->p8Resize(m_type0_p, t_p8, inv_rate, inv_rate);
                    p_config->___invariant.push_back(t_p8);
                }
                if (i_r_type0_p == 8)
                {
                    i_r_type0_p = 0;
                    p_mask_maker->p8Resize(r_type0_p, t_p8, inv_rate, inv_rate);
                    p_config->___erase_area.push_back(t_p8);
                }
            }
            if (is_preview)
            {
                p_mask_maker->preview_mask(*p_config, (*tb_5_value) / 100.0, *p_effect_mat);
            }
            break;
        }
        case 1:
        {
            if (event == CV_EVENT_LBUTTONDOWN)//左键按下，读取初始坐标  
            {
                is_l_down_type1 = true;
                l_type1_point[0].x = x;
                l_type1_point[0].y = y;
            }
            if (event == CV_EVENT_MBUTTONDOWN)//中间按下，读取初始坐标 
            {
                is_m_down_type1 = true;
                m_type1_point[0].x = x;
                m_type1_point[0].y = y;
            }
            if (event == CV_EVENT_RBUTTONDOWN)//右键按下，读取初始坐标  
            {
                is_r_down_type1 = true;
                r_type1_point[0].x = x;
                r_type1_point[0].y = y;
            }
            if (event == CV_EVENT_LBUTTONUP || event == CV_EVENT_MBUTTONUP || event == CV_EVENT_RBUTTONUP)//按键松开  
            {
                is_l_down_type1 = false;
                is_m_down_type1 = false;
                is_r_down_type1 = false;
                cv::Mat org_img = cv::imread(p_config->obj_path, CV_LOAD_IMAGE_UNCHANGED);
                if (event == CV_EVENT_LBUTTONUP)
                {
                    l_type1_point[1].x = x;
                    l_type1_point[1].y = y;

                    int rect_x, rect_y, rect_width, rect_height;
                    if (l_type1_point[1].x < l_type1_point[0].x) { rect_x = l_type1_point[1].x; rect_width = l_type1_point[0].x - l_type1_point[1].x; }
                    else { rect_x = l_type1_point[0].x; rect_width = l_type1_point[1].x - l_type1_point[0].x; }
                    if (l_type1_point[1].y < l_type1_point[0].y) { rect_y = l_type1_point[1].y; rect_height = l_type1_point[0].y - l_type1_point[1].y; }
                    else { rect_y = l_type1_point[0].y; rect_height = l_type1_point[1].y - l_type1_point[0].y; }

                    double inv_rate = sqrt(double(org_img.cols) * org_img.rows / rate_param);
                    rect_x *= inv_rate;
                    rect_y *= inv_rate;
                    rect_width *= inv_rate;
                    rect_height *= inv_rate;

                    if (rect_width <= 5 || rect_height <= 5) { return; }
                    if (rect_x < 0) { rect_x = 0; }
                    if (rect_y < 0) { rect_y = 0; }
                    if (rect_width + rect_x > org_img.cols) { rect_width = org_img.cols - rect_x; }
                    if (rect_height + rect_y > org_img.rows) { rect_height = org_img.rows - rect_y; }

                    std::vector<int> erase_index;
                    cv::Rect del_rect(rect_x, rect_y, rect_width, rect_height);
                    for (int i = 0; i < p_config->___transparent.size(); ++i)
                    {
                        for(int j=0;j<8;++j)
                        {
                            if(del_rect.contains(p_config->___transparent[i][j]))
                            {
                                erase_index.push_back(i);
                                break;
                            }
                        }
                    }
                    for (int i = erase_index.size() - 1; i >= 0; --i)
                    {
                        p_config->___transparent.erase(p_config->___transparent.begin() + erase_index[i]);
                    }
                }
                else if (event == CV_EVENT_MBUTTONUP)
                {
                    m_type1_point[1].x = x;
                    m_type1_point[1].y = y;

                    int rect_x, rect_y, rect_width, rect_height;
                    if (m_type1_point[1].x < m_type1_point[0].x) { rect_x = m_type1_point[1].x; rect_width = m_type1_point[0].x - m_type1_point[1].x; }
                    else { rect_x = m_type1_point[0].x; rect_width = m_type1_point[1].x - m_type1_point[0].x; }
                    if (m_type1_point[1].y < m_type1_point[0].y) { rect_y = m_type1_point[1].y; rect_height = m_type1_point[0].y - m_type1_point[1].y; }
                    else { rect_y = m_type1_point[0].y; rect_height = m_type1_point[1].y - m_type1_point[0].y; }

                    double inv_rate = sqrt(double(org_img.cols) * org_img.rows / rate_param);
                    rect_x *= inv_rate;
                    rect_y *= inv_rate;
                    rect_width *= inv_rate;
                    rect_height *= inv_rate;

                    if (rect_width <= 5 || rect_height <= 5) { return; }

                    if (rect_x < 0) { rect_x = 0; }
                    if (rect_y < 0) { rect_y = 0; }
                    if (rect_width + rect_x > org_img.cols) { rect_width = org_img.cols - rect_x; }
                    if (rect_height + rect_y > org_img.rows) { rect_height = org_img.rows - rect_y; }

                    std::vector<int> erase_index;
                    cv::Rect del_rect(rect_x, rect_y, rect_width, rect_height);

                    for (int i = 0; i < p_config->___invariant.size(); ++i)
                    {
                        for (int j = 0; j<8; ++j)
                        {
                            if (del_rect.contains(p_config->___invariant[i][j]))
                            {
                                erase_index.push_back(i);
                                break;
                            }
                        }
                    }
                    for (int i = erase_index.size() - 1; i >= 0; --i)
                    {
                        p_config->___invariant.erase(p_config->___invariant.begin() + erase_index[i]);
                    }
                }
                else if (event == CV_EVENT_RBUTTONUP)
                {
                    r_type1_point[1].x = x;
                    r_type1_point[1].y = y;

                    int rect_x, rect_y, rect_width, rect_height;
                    if (r_type1_point[1].x < r_type1_point[0].x) { rect_x = r_type1_point[1].x; rect_width = r_type1_point[0].x - r_type1_point[1].x; }
                    else { rect_x = r_type1_point[0].x; rect_width = r_type1_point[1].x - r_type1_point[0].x; }
                    if (r_type1_point[1].y < r_type1_point[0].y) { rect_y = r_type1_point[1].y; rect_height = r_type1_point[0].y - r_type1_point[1].y; }
                    else { rect_y = r_type1_point[0].y; rect_height = r_type1_point[1].y - r_type1_point[0].y; }

                    double inv_rate = sqrt(double(org_img.cols) * org_img.rows / rate_param);
                    rect_x *= inv_rate;
                    rect_y *= inv_rate;
                    rect_width *= inv_rate;
                    rect_height *= inv_rate;

                    if (rect_width <= 5 || rect_height <= 5) { return; }

                    if (rect_x < 0) { rect_x = 0; }
                    if (rect_y < 0) { rect_y = 0; }
                    if (rect_width + rect_x > org_img.cols) { rect_width = org_img.cols - rect_x; }
                    if (rect_height + rect_y > org_img.rows) { rect_height = org_img.rows - rect_y; }

                    std::vector<int> erase_index;
                    cv::Rect del_rect(rect_x, rect_y, rect_width, rect_height);

                    for (int i = 0; i < p_config->___erase_area.size(); ++i)
                    {
                        for (int j = 0; j < 8; ++j)
                        {
                            if (del_rect.contains(p_config->___erase_area[i][j]))
                            {
                                erase_index.push_back(i);
                                break;
                            }
                        }
                    }
                    for (int i = erase_index.size() - 1; i >= 0; --i)
                    {
                        p_config->___erase_area.erase(p_config->___erase_area.begin() + erase_index[i]);
                    }
                }
                org_img.release();
                p_mask_maker->preview_mask(*p_config, (*tb_5_value) / 100.0, *p_effect_mat);
            }
            break;
        }
        }
    }
    return;
}

int mask_maker::makeConfigGUI(mask_config &inputOutput_config)
{
    static cv::Mat effect_mat;
    static int tb_value[6] = { 0,0,0,0,0,100 };
    tb_value[0] = inputOutput_config.param.at(CONFIG_KEY_bg_threshold);
    tb_value[1] = inputOutput_config.param.at(CONFIG_KEY_fg_threshold);
    tb_value[2] = inputOutput_config.param.at(CONFIG_KEY_edge_smooth_kernal_size);
    tb_value[3] = inputOutput_config.param.at(CONFIG_KEY_mask_threshold);

    if (tb_value[0] > 50)
    {
        tb_value[0] = 49.0 / 204.0*(tb_value[0] - 255) + 100;
        if (tb_value[0] > 100) { tb_value[0] = 100; }
        else if (tb_value[0] < 0) { tb_value[0] = 0; }
    }

    cv::namedWindow("抠图参数配置器");
    void *(bar_pack[4]);
    bar_pack[0] = this;
    bar_pack[1] = &inputOutput_config;
    bar_pack[2] = &(tb_value[5]);
    bar_pack[3] = &effect_mat;
    cv::createTrackbar("背景阈值", "抠图参数配置器",
        &tb_value[0], 100, on_Trackbar_config_value_0, bar_pack);
    cv::createTrackbar("前景阈值", "抠图参数配置器",
        &tb_value[1], 30, on_Trackbar_config_value_1, bar_pack);
    cv::createTrackbar("边缘光滑值", "抠图参数配置器",
        &tb_value[2], 100, on_Trackbar_config_value_2, bar_pack);
    cv::createTrackbar("maks大小调整", "抠图参数配置器",
        &tb_value[3], 100, on_Trackbar_config_value_3, bar_pack);
    cv::createTrackbar("增删选框", "抠图参数配置器",
        &tb_value[4], 1);
    cv::createTrackbar("预览调整", "抠图参数配置器",
        &tb_value[5], 100, on_Trackbar_config_value_4, bar_pack);

    void *(btn_pack[6]);
    btn_pack[0] = this;
    btn_pack[1] = &inputOutput_config;
    btn_pack[2] = &(tb_value[4]);
    btn_pack[3] = &(tb_value[5]);
    btn_pack[4] = &effect_mat;
    bool is_shift_obj = true;
    btn_pack[5] = &is_shift_obj;
    cv::setMouseCallback("抠图参数配置器", on_Mouse, btn_pack);
    if (!preview_mask(inputOutput_config, tb_value[5]/100.0, effect_mat))
    {
        return 27;//"ESC"
    }
    int key_value = cv::waitKey(0);
    return key_value;
}

mask_maker::mask_maker(const std::string &_config_save_path)
{
    set_config_save_path(_config_save_path);

    //TODO: to initialize config_default only once;
    config_default.param[CONFIG_KEY_bg_threshold] = 15;
    config_default.param[CONFIG_KEY_fg_threshold] = 2;
    config_default.param[CONFIG_KEY_edge_smooth_kernal_size] = 5;
    config_default.param[CONFIG_KEY_mask_threshold] = 50;
}

void mask_maker::set_config_save_path(const std::string &_config_save_path)
{
    ___config_save_path = _config_save_path;
}

bool mask_maker::str2point8(const std::string &config_str, std::vector<cv::Point> &P8)
{
    std::vector<std::string> p_str;
    std::expand::split(config_str, ".", p_str);
    if (p_str.size() != 16) return false;
    P8.resize(8);
    try
    {
        for (int i = 0; i < 8; ++i)
        {
            P8[i].x = std::stoi(p_str[i * 2]);
            P8[i].y = std::stoi(p_str[i * 2 + 1]);
        }
    }
    catch (...)
    {
        std::cout << "mask_maker: str2point8: std::stoi error." << std::endl;
        return false;
    }
    return true;
}

bool mask_maker::point82str(const std::vector<cv::Point> &P8, std::string &config_str)
{
    try
    {
        config_str = "";
        for (int i = 0; i < 8; ++i)
        {
            config_str += std::to_string(P8[i].x);
            config_str += ".";
            config_str += std::to_string(P8[i].y);
            config_str += ".";
        }
    }
    catch (...)
    {
        std::cout << "mask_maker: point82str: std::to_string error." << std::endl;
        return false;
    }
    return true;
}

cv::Mat mask_maker::p8AreaMask(const cv::Size &src_size, const std::vector<std::vector<cv::Point>> &areas)
{
    cv::Mat output(src_size, CV_8UC1);
    output.setTo(0);
    for(int i = 0;i<areas.size();++i)
    {
        cv::fillConvexPoly(output, areas[i], 255);
    }
    return output;
}

cv::Mat mask_maker::p8AreaInvMask(const cv::Size &src_size, const std::vector<std::vector<cv::Point>> &areas)
{
    cv::Mat output = p8AreaMask(src_size, areas);
    cv::threshold(output, output, 127, 255, CV_THRESH_BINARY_INV);
    return output;
}

void mask_maker::p8Resize(const std::vector<cv::Point> &src, std::vector<cv::Point> &dst, double x_rate, double y_rate)
{
    dst.resize(8);
    for (int i = 0; i < 8; ++i)
    {
        dst[i].x = x_rate*src[i].x;
        dst[i].y = y_rate*src[i].y;
    }
    return;
}

//bool mask_maker::str2rect(const std::string &config_str, cv::Rect &rect)
//{
//    //x0y0w100h50
//    std::vector<std::string> rect_str;
//    std::expand::split(config_str, "xywh", rect_str);
//    if (rect_str.size() != 4) return false;
//    try
//    {
//        rect.x = std::stoi(rect_str[0]);
//        rect.y = std::stoi(rect_str[1]);
//        rect.width = std::stoi(rect_str[2]);
//        rect.height = std::stoi(rect_str[3]);
//    }
//    catch (...)
//    {
//        std::cout << "mask_maker: str2rect: std::stoi error." << std::endl;
//        return false;
//    }
//    return true;
//}
//
//bool mask_maker::rect2str(const cv::Rect &rect, std::string &config_str)
//{
//    //x0y0w100h50
//    try {
//        config_str = 'x';
//        config_str += std::to_string(rect.x);
//        config_str += 'y';
//        config_str += std::to_string(rect.y);
//        config_str += 'w';
//        config_str += std::to_string(rect.width);
//        config_str += 'h';
//        config_str += std::to_string(rect.height);
//    }
//    catch (...)
//    {
//        std::cout << "mask_maker: rect2str: std::to_string error." << std::endl;
//        return false;
//    }
//    return true;
//}


bool mask_maker::make_config(const std::vector<std::vector<std::string>> &obj_paths, std::vector<std::vector<mask_config>> &config, const std::vector<mask_config> &loaded_config)
{
    int key_value;
    int obj_second = 0;
    int obj_first = 0;

    config.resize(obj_paths.size());

    std::unordered_map<std::string, int > map_loaded_config;
    for (int i = 0; i < loaded_config.size(); ++i)
    {
        map_loaded_config[loaded_config[i].obj_path] = i;
    }

    for (int i = 0; i < obj_paths.size(); ++i)
    {
        config[i].resize(obj_paths[i].size());
        for (int j = 0; j < obj_paths[i].size(); ++j)
        {
            auto index_iter = map_loaded_config.find(obj_paths[i][j]);
            if (index_iter == map_loaded_config.end())
            {
                config[i][j] = config_default;
                config[i][j].obj_path = obj_paths[i][j];
            }
            else
            {
                config[i][j] = loaded_config[index_iter->second];
            }
        }
    }

    while (1)
    {

        key_value = makeConfigGUI(config[obj_first][obj_second]);
        //std::cout << config[obj_first][obj_second].transparent.size() << "    "
        //    << config[obj_first][obj_second].invariant.size()
        //    << std::endl;
        if (key_value == 'd' || key_value == 'D')
        {
            int t_size = config[obj_first].size();
            //for (int i = 0; i < t_size; ++i)
            //{
            //    config[obj_first][i].param = config[obj_first][obj_second].param;
            //}
            if (obj_second + 1 == t_size)
            {
                obj_first = (obj_first + 1 + config.size()) % config.size(); obj_second = 0;
            }
            else
            {
                obj_second = (obj_second + 1 + t_size) % t_size;
            }
        }

        else if (key_value == 'a' || key_value == 'A')
        {
            int t_size = config[obj_first].size();
            //for (int i = 0; i < t_size; ++i)
            //{
            //    config[obj_first][i].param = config[obj_first][obj_second].param;
            //}
            if (obj_second  == 0)
            {
                obj_first = (obj_first - 1 + config.size()) % config.size();
                obj_second = t_size - 1;
            }
            else
            {
                obj_second = (obj_second - 1 + t_size) % t_size;
            }

        }
        else if (key_value == 'w' || key_value == 'W') { obj_first = (obj_first - 1 + config.size()) % config.size(); obj_second = 0; }
        else if (key_value == 's' || key_value == 'S') { obj_first = (obj_first + 1 + config.size()) % config.size(); obj_second = 0; }
        else if (key_value == '\r' || key_value == 'y' || key_value == 'Y')
        {
            cv::destroyAllWindows();
            return true;
        }
        else if (key_value == 27)
        {
            std::cout << "quit" << std::endl;
            cv::destroyAllWindows();
            return false;
        }
    }
}
//
//void mask_maker::make_config(std::vector<std::string> obj_paths, std::vector<mask_config> &config)
//{
//    if()
//    int i = 0;
//
//
//}


bool mask_maker::load_config(const std::string &config_path, std::vector<mask_config> &config)
{
    std::ifstream file(config_path);
    if (!file.is_open())
    {
        std::cout << "mask_maker: load_config: can not find mask config file." << std::endl;
        return false;
    }
    std::vector<std::string> config_list;

    while (!file.eof())
    {
        std::string one_line;
        std::getline(file, one_line);
        if (one_line.size() == 0) continue;
        one_line = one_line.substr(0, one_line.find_first_of('#'));
        if (one_line.size() == 0) continue;
        config_list.push_back(one_line);
    }

    config.resize(0);
    if (config_list.size() == 0) 
    {
        std::cout << "mask_maker: load_config: No mask config." << std::endl;
        return false;
    }

    po::options_description one_desc("one config");
    //std::string _obj_path;
    int _bg_threshold;
    int _fg_threshold;
    int _edge_smooth_kernal_size;
    int _mask_threshold;
    std::vector<std::string> _transparent_str;
    std::vector<std::string> _invariant_str;
    std::vector<std::string> _erase_area_str;
    one_desc.add_options()
        //("obj_path", po::value<std::string>(&_obj_path), "")
        ("bg_threshold", po::value<int>(&_bg_threshold)->default_value(config_default.param.at(CONFIG_KEY_bg_threshold)), "")
        ("fg_threshold", po::value<int>(&_fg_threshold)->default_value(config_default.param.at(CONFIG_KEY_fg_threshold)), "")
        ("edge_smooth_kernal_size", po::value<int>(&_edge_smooth_kernal_size)->default_value(config_default.param.at(CONFIG_KEY_edge_smooth_kernal_size)), "")
        ("mask_threshold", po::value<int>(&_mask_threshold)->default_value(config_default.param.at(CONFIG_KEY_mask_threshold)), "")
        ("transparent", po::value<std::vector<std::string>>(&_transparent_str)->multitoken(), "")
        ("invariant", po::value<std::vector<std::string>>(&_invariant_str)->multitoken(), "")
        ("erase_area", po::value<std::vector<std::string>>(&_erase_area_str)->multitoken(), "");

    for (int i = 0; i < config_list.size(); ++i)
    {
        //default;
        _bg_threshold = config_default.param[CONFIG_KEY_bg_threshold];
        _fg_threshold = config_default.param[CONFIG_KEY_fg_threshold];
        _edge_smooth_kernal_size = config_default.param[CONFIG_KEY_edge_smooth_kernal_size];
        _mask_threshold = config_default.param[CONFIG_KEY_mask_threshold];
        _transparent_str.resize(0);
        _invariant_str.resize(0);
        _erase_area_str.resize(0);

        po::variables_map one_config_vm;
        size_t index = config_list[i].find(" --");
        std::vector<std::string> _part;
        std::string _obj_path;
        if (index != std::string::npos)
        {
            _obj_path = config_list[i].substr(0, index);
            while (_obj_path.size() > 0 && _obj_path[_obj_path.size() - 1] == ' ') { _obj_path.pop_back(); }
            std::string _rest_part = config_list[i].substr(index);
            std::expand::split(_rest_part, " ", _part);
        }
        else
        {
            _obj_path = config_list[i];
            while (_obj_path.size() > 0 && _obj_path[_obj_path.size() - 1] == ' ') { _obj_path.pop_back(); }
        }
        int argc = _part.size() + 1;
        char const **argv = new const char *[_part.size() + 1];
        argv[0] = _obj_path.c_str();
        for (int m = 0; m < _part.size(); ++m) { argv[m + 1] = _part[m].c_str(); }
        try
        {
            po::store(po::parse_command_line(argc, argv, one_desc), one_config_vm);
            po::notify(one_config_vm);
        }
        catch (boost::program_options::error_with_no_option_name &ex)
        {
            std::cout << ex.what() << std::endl;
            return false;
        }
        delete[] argv;
        argv = NULL;

        mask_config one_config;
        one_config.obj_path = _obj_path;
        one_config.param[CONFIG_KEY_bg_threshold] = _bg_threshold;
        one_config.param[CONFIG_KEY_fg_threshold] = _fg_threshold;
        one_config.param[CONFIG_KEY_edge_smooth_kernal_size] = _edge_smooth_kernal_size;
        one_config.param[CONFIG_KEY_mask_threshold] = _mask_threshold;

        std::vector<cv::Point> t_point8;
        one_config.___transparent.resize(0);
        for (int m = 0; m < _transparent_str.size(); ++m)
        {
            if (!str2point8(_transparent_str[m], t_point8))
            {
                std::cout << _transparent_str[m] << std::endl;
                return false;
            }
            else
            {
                one_config.___transparent.push_back(t_point8);
            }
        }
        one_config.___invariant.resize(0);
        for (int m = 0; m < _invariant_str.size(); ++m)
        {
            if (!str2point8(_invariant_str[m], t_point8))
            {
                return false;
            }
            else
            {
                one_config.___invariant.push_back(t_point8);
            }
        }
        one_config.___erase_area.resize(0);
        for (int m = 0; m < _erase_area_str.size(); ++m)
        {
            if (!str2point8(_erase_area_str[m], t_point8))
            {
                return false;
            }
            else
            {
                one_config.___erase_area.push_back(t_point8);
            }
        }

        //cv::Rect t_rect;
        //one_config.transparent.resize(0);
        //for (int m = 0; m < _transparent_str.size(); ++m)
        //{
        //    if (!str2rect(_transparent_str[m], t_rect))
        //    {
        //        std::cout << _transparent_str[m] << std::endl;
        //        return false;
        //    }
        //    else
        //    {
        //        one_config.transparent.push_back(t_rect);
        //    }
        //}
        //one_config.invariant.resize(0);
        //for (int m = 0; m < _invariant_str.size(); ++m)
        //{
        //    if (!str2rect(_invariant_str[m], t_rect))
        //    {
        //        return false;
        //    }
        //    else
        //    {
        //        one_config.invariant.push_back(t_rect);
        //    }
        //}
        //one_config.erase_area.resize(0);
        //for (int m = 0; m < _erase_area_str.size(); ++m)
        //{
        //    if (!str2rect(_erase_area_str[m], t_rect))
        //    {
        //        return false;
        //    }
        //    else
        //    {
        //        one_config.erase_area.push_back(t_rect);
        //    }
        //}
        config.push_back(one_config);
    }
    return true;
}

bool mask_maker::save_config(const std::vector<std::vector<mask_config>> &config)
{
    size_t index = ___config_save_path.find_last_of("\\/");
    if (index != std::string::npos) 
    {
        if (!std::expand::create_directory(___config_save_path.substr(0, index)))
        {
            std::cout << "mask_maker: save_config: can not create directory: " << std::endl;
            std::cout << ___config_save_path.substr(0, index) << std::endl;
            return false;
        }
    }
    std::ofstream file(___config_save_path);
    if (!file.is_open()) 
    { 
        std::cout << "mask_maker: save_config: can not create mask config file." << std::endl;
        return false; }
    for (int i = 0; i < config.size(); ++i)
    {
        for (int j = 0; j < config[i].size(); ++j)
        {
            file << config[i][j].obj_path;
            file << " " << "--bg_threshold=" << config[i][j].param.at(CONFIG_KEY_bg_threshold);
            file << " " << "--fg_threshold=" << config[i][j].param.at(CONFIG_KEY_fg_threshold);
            file << " " << "--edge_smooth_kernal_size=" << config[i][j].param.at(CONFIG_KEY_edge_smooth_kernal_size);
            file << " " << "--mask_threshold=" << config[i][j].param.at(CONFIG_KEY_mask_threshold);
            std::string t_str;
            for (int m = 0; m < config[i][j].___transparent.size(); ++m)
            {
                if (!point82str(config[i][j].___transparent[m], t_str)) { return false; }
                file << " --transparent=" << t_str;
            }
            for (int m = 0; m < config[i][j].___invariant.size(); ++m)
            {
                if (!point82str(config[i][j].___invariant[m], t_str)) { return false; }
                file << " --invariant=" << t_str;
            }
            for (int m = 0; m < config[i][j].___erase_area.size(); ++m)
            {
                if (!point82str(config[i][j].___erase_area[m], t_str)) { return false; }
                file << " --erase_area=" << t_str;
            }

            //for (int m = 0; m < config[i][j].transparent.size(); ++m)
            //{
            //    if (!rect2str(config[i][j].transparent[m], t_str)) { return false; }
            //    file << " --transparent=" << t_str;
            //}
            //for (int m = 0; m < config[i][j].invariant.size(); ++m)
            //{
            //    if (!rect2str(config[i][j].invariant[m], t_str)) { return false; }
            //    file << " --invariant=" << t_str;
            //}
            //for (int m = 0; m < config[i][j].erase_area.size(); ++m)
            //{
            //    if (!rect2str(config[i][j].erase_area[m], t_str)) { return false; }
            //    file << " --erase_area=" << t_str;
            //}
            file << std::endl;
        }
    }
    return true;
}

bool mask_maker::save_config(const std::vector<mask_config> &config)
{
    std::vector<std::vector<mask_config>> _config;
    _config.push_back(config);
    return save_config(_config);
}



cv::Mat mask_maker::make_mask(const mask_config &config)
{
    return Matting(config);
}

void mask_maker::imshow_png(const std::string &windowsName, const cv::Mat &mat_8uc4,double bg_brightness)
{
    cv::Mat pattern = getInvariantPattern(mat_8uc4);

    cv::Mat _to_show(mat_8uc4.rows, mat_8uc4.cols, CV_8UC3);

    //auto iter_end_mat_8uc4 = mat_8uc4.end<cv::Vec4b>();
    //auto iter_mat_8uc4 = mat_8uc4.begin<cv::Vec4b>();
    //auto iter_pattern = pattern.begin<uchar>();
    //auto iter_to_show = _to_show.begin<cv::Vec3b>();

    //for (;
    //    iter_mat_8uc4 != iter_end_mat_8uc4;
    //    ++iter_mat_8uc4, ++iter_to_show, ++iter_pattern)
    //{
    //    float  weight = (*iter_mat_8uc4)[3] / 255.0;
    //    (*iter_to_show)[0] = (*iter_mat_8uc4)[0] * weight +(*iter_pattern)*(1 - weight)*bg_brightness;
    //    (*iter_to_show)[1] = (*iter_mat_8uc4)[1] * weight +(*iter_pattern)*(1 - weight)*bg_brightness;
    //    (*iter_to_show)[2] = (*iter_mat_8uc4)[2] * weight +(*iter_pattern)*(1 - weight)*bg_brightness;
    //}

    for (int i = 0; i < mat_8uc4.rows; ++i)
    {
        const cv::Vec4b *mat_8uc4_line = mat_8uc4.ptr<cv::Vec4b>(i);
        const uchar *pattern_line = pattern.ptr<uchar>(i);
        cv::Vec3b *_to_show_line = _to_show.ptr<cv::Vec3b>(i);
        for (int j = 0; j < mat_8uc4.cols; ++j)
        {
            float  weight = mat_8uc4_line[j][3] / 255.0;
            _to_show_line[j][0] = mat_8uc4_line[j][0] * weight + pattern_line[j] * (1 - weight)*bg_brightness;
            _to_show_line[j][1] = mat_8uc4_line[j][1] * weight + pattern_line[j] * (1 - weight)*bg_brightness;
            _to_show_line[j][2] = mat_8uc4_line[j][2] * weight + pattern_line[j] * (1 - weight)*bg_brightness;
        }
    }

    cv::imshow(windowsName, _to_show);
    return;
}

//
//
//
//#include"mask_configurator.h"
//#include"mask_maker.h"
//
//#include <boost/program_options.hpp>  
//
//
//#include<opencv2\highgui\highgui.hpp>
//#include<opencv2\imgproc\imgproc.hpp>
//
//#include<iostream>
//#include<string>
//#include<vector>
//
//#include<direct.h>
//#include<fstream>
//
//
//namespace po = boost::program_options;
//
//cv::Mat mask_configurator::obj;
//std::string mask_configurator::config_file_save_path = "MASK_CONFIG";
//
//void mask_configurator::on_Trackbar_config_value_1_0(int bar_value, void *v_p_config)
//{
//    std::unordered_map<std::string, int> * p_config = (std::unordered_map<std::string, int> *)v_p_config;
//    if (bar_value > 50)
//    {
//        bar_value = 204.0 / 49.0*(bar_value - 100) + 255;
//        if (bar_value > 255) { bar_value = 255; }
//        else if (bar_value < 0) { bar_value = 0; }
//    }
//    (*p_config)["bg_threshold"] = bar_value;
//    previewMask(*p_config);
//    return;
//}
//
//void mask_configurator::on_Trackbar_config_value_1_1(int bar_value, void *v_p_config)
//{
//    std::unordered_map<std::string, int> * p_config = (std::unordered_map<std::string, int> *)v_p_config;
//    (*p_config)["fg_threshold"] = bar_value;
//    previewMask(*p_config);
//    return;
//}
//
//void mask_configurator::on_Trackbar_config_value_1_2(int bar_value, void *v_p_config)
//{
//    std::unordered_map<std::string, int> * p_config = (std::unordered_map<std::string, int> *)v_p_config;
//    (*p_config)["edge_smooth_kernal_size"] = bar_value;
//    previewMask(*p_config);
//    return;
//}
//
//void mask_configurator::on_Trackbar_config_value_1_3(int bar_value, void *v_p_config)
//{
//    std::unordered_map<std::string, int> * p_config = (std::unordered_map<std::string, int> *)v_p_config;
//    (*p_config)["mask_threshold"] = bar_value;
//    previewMask(*p_config);
//    return;
//}
//
//void mask_configurator::setConfigFilePath(const std::string &_path)
//{
//    config_file_save_path = _path;
//    return;
//}
//
//bool mask_configurator::getDefaultConfig(std::unordered_map<std::string, int> &config)
//{
//    config.clear();
//    config["make_type"] = GLOBAL_MATTING;
//    config["bg_threshold"] = 30;
//    config["fg_threshold"] = 3;
//    config["edge_smooth_kernal_size"] = 7;
//    config["mask_threshold"] = 50;
//    return true;
//}
//
//bool mask_configurator::tryReadConfig(std::unordered_map<std::string, int> &config)
//{
//    //open file
//    std::ifstream mask_config_file(config_file_save_path);
//    if (!mask_config_file.is_open())
//    {
//        std::cout << "No mask config file.Use default." << std::endl;
//        getDefaultConfig(config);
//        mask_config_file.close();
//        mask_config_file.clear();
//        return false;
//    }
//
//    //analysis config file 
//
//    std::string make_type;
//    int bg_threshold;
//    int fg_threshold;
//    int edge_smooth_kernal_size;
//    int mask_threshold;
//    po::options_description all("mask configurator options");
//    all.add_options()
//        ("make_type", po::value<std::string>(&make_type)->default_value("GLOBAL_MATTING"), "NO_PROCESS  FRAME_DIFFERENCE  GLOBAL_MATTING.")
//        ("bg_threshold", po::value<int>(&bg_threshold)->default_value(30), "Blackground threshold.")
//        ("fg_threshold", po::value<int>(&fg_threshold)->default_value(3), "Frontground threshold.")
//        ("edge_smooth_kernal_size", po::value<int>(&edge_smooth_kernal_size)->default_value(7), "Edge smooth kernal size.")
//        ("mask_threshold", po::value<int>(&mask_threshold)->default_value(50), "Mask threshold.");
//
//    po::variables_map readed_mask_config_param;
//
//    //po::store(po::parse_config_file(mask_config_file, all), readed_mask_config_param);
//    //po::notify(readed_mask_config_param);
//
//
//    try
//    {
//        po::store(po::parse_config_file(mask_config_file, all), readed_mask_config_param);
//        po::notify(readed_mask_config_param);
//    }
//    catch (boost::program_options::error_with_no_option_name &ex)
//    {
//        std::cout << all << std::endl;
//        std::cout << ex.what() << std::endl;
//        std::cout << "Read mask config file error.Use default." << std::endl;
//        getDefaultConfig(config);
//        return false;
//    }
//
//    //set config
//    if (readed_mask_config_param.count("make_type"))
//    {
//        if (make_type == "GLOBAL_MATTING") config["make_type"] = GLOBAL_MATTING;
//        else config["make_type"] = UNKNOWN;
//    }
//    else
//    {
//        config["make_type"] = UNKNOWN;
//    }
//
//    if (readed_mask_config_param.count("bg_threshold"))
//    {
//        config["bg_threshold"] = bg_threshold;
//    }
//
//    if (readed_mask_config_param.count("fg_threshold"))
//    {
//        config["fg_threshold"] = fg_threshold;
//    }
//    if (readed_mask_config_param.count("edge_smooth_kernal_size"))
//    {
//        config["edge_smooth_kernal_size"] = edge_smooth_kernal_size;
//    }
//    if (readed_mask_config_param.count("mask_threshold"))
//    {
//        config["mask_threshold"] = mask_threshold;
//    }
//
//    return true;
//
//}
//
//bool mask_configurator::tryReadConfig(std::unordered_map<std::string, int> &config,const std::string _config_file_save_path)
//{
//    //open file
//    std::ifstream mask_config_file(_config_file_save_path);
//    if (!mask_config_file.is_open())
//    {
//        std::cout << "No mask config file.Use default." << std::endl;
//        getDefaultConfig(config);
//        mask_config_file.close();
//        mask_config_file.clear();
//        return false;
//    }
//
//    //analysis config file 
//
//    std::string make_type;
//    int bg_threshold;
//    int fg_threshold;
//    int edge_smooth_kernal_size;
//    int mask_threshold;
//    po::options_description all("mask configurator options");
//    all.add_options()
//        ("make_type", po::value<std::string>(&make_type)->default_value("GLOBAL_MATTING"), "NO_PROCESS  FRAME_DIFFERENCE  GLOBAL_MATTING.")
//        ("bg_threshold", po::value<int>(&bg_threshold)->default_value(30), "Blackground threshold.")
//        ("fg_threshold", po::value<int>(&fg_threshold)->default_value(3), "Frontground threshold.")
//        ("edge_smooth_kernal_size", po::value<int>(&edge_smooth_kernal_size)->default_value(7), "Edge smooth kernal size.")
//        ("mask_threshold", po::value<int>(&mask_threshold)->default_value(50), "Mask threshold.");
//
//    po::variables_map readed_mask_config_param;
//
//    //po::store(po::parse_config_file(mask_config_file, all), readed_mask_config_param);
//    //po::notify(readed_mask_config_param);
//
//
//    try
//    {
//        po::store(po::parse_config_file(mask_config_file, all), readed_mask_config_param);
//        po::notify(readed_mask_config_param);
//    }
//    catch (boost::program_options::error_with_no_option_name &ex)
//    {
//        std::cout << all << std::endl;
//        std::cout << ex.what() << std::endl;
//        std::cout << "Read mask config file error.Use default." << std::endl;
//        getDefaultConfig(config);
//        return false;
//    }
//
//    //set config
//    if (readed_mask_config_param.count("make_type"))
//    {
//        if (make_type == "GLOBAL_MATTING") config["make_type"] = GLOBAL_MATTING;
//        else config["make_type"] = UNKNOWN;
//    }
//    else
//    {
//        config["make_type"] = UNKNOWN;
//    }
//
//    if (readed_mask_config_param.count("bg_threshold"))
//    {
//        config["bg_threshold"] = bg_threshold;
//    }
//
//    if (readed_mask_config_param.count("fg_threshold"))
//    {
//        config["fg_threshold"] = fg_threshold;
//    }
//    if (readed_mask_config_param.count("edge_smooth_kernal_size"))
//    {
//        config["edge_smooth_kernal_size"] = edge_smooth_kernal_size;
//    }
//    if (readed_mask_config_param.count("mask_threshold"))
//    {
//        config["mask_threshold"] = mask_threshold;
//    }
//
//    return true;
//
//}
//
//
//bool mask_configurator::checkConfig(std::unordered_map<std::string, int> &config)
//{
//    bool flag = false;
//    if (config.find("make_type") != config.end()) 
//    {
//        switch (config["make_type"])
//        {
//        case NO_PROCESS:
//        {
//            flag = true;
//            break;
//        }
//        case FRAME_DIFFERENCE:
//        {
//            flag = true;
//            break;
//        }
//        case GLOBAL_MATTING:
//        {
//            if (config.find("bg_threshold") != config.end()
//                && config.find("fg_threshold") != config.end()
//                && config.find("edge_smooth_kernal_size") != config.end()
//                && config.find("mask_threshold") != config.end())
//                if (0 <= config.at("bg_threshold") && config.at("bg_threshold") <= 255)
//                    if (0 <= config.at("fg_threshold") && config.at("fg_threshold") <= 30)
//                        if (0 <= config.at("edge_smooth_kernal_size") && config.at("edge_smooth_kernal_size") <= 30)
//                            if (0 <= config.at("mask_threshold") && config.at("mask_threshold") <= 100)
//                                flag = true;
//            break;
//        }
//        default:
//        {
//            break;
//        }
//        }
//    }
//    //default
//    if (!flag)
//    {
//        getDefaultConfig(config);
//    }
//
//    return flag;
//}
//
//bool mask_configurator::previewNoProcess(std::unordered_map<std::string, int> &inputOutput_config)
//{
//    cv::Mat mask = mask_maker::createMask(obj, inputOutput_config);
//    if (mask.empty()) return false;//"ESC"
//
//    double rate = 1;
//    rate = sqrt(750.0 * 500.0 / mask.cols / mask.rows);
//    cv::Size fit_size(mask.cols*rate, mask.rows*rate);
//    cv::Mat fit_obj;
//    cv::resize(obj, fit_obj, fit_size);
//
//    for (int i = 0; i < fit_obj.rows; ++i)
//    {
//        for (int j = 0; j < fit_obj.cols; ++j)
//        {
//            if (mask.at<uchar>(i, j) == 0)
//            {
//                fit_obj.at<cv::Vec3b>(i, j)[0] *= 0.3;
//                fit_obj.at<cv::Vec3b>(i, j)[1] *= 0.3;
//                fit_obj.at<cv::Vec3b>(i, j)[2] *= 0.3;
//            }
//        }
//    }
//    imshow("type - 0 - mask previewer", fit_obj);
//    return true;
//}
//
//bool mask_configurator::previewFrameDifference(std::unordered_map<std::string, int> &inputOutput_config)
//{
//    return false;//"ESC"
//}
//
//bool mask_configurator::previewGlobalMatting(std::unordered_map<std::string, int> &inputOutput_config)
//{
//    std::vector<cv::Mat> middle_mat;
//    cv::Mat mask = mask_maker::createMask(obj, inputOutput_config, middle_mat);
//    if (mask.empty()) return false;//"ESC"
//
//    double rate = 1;
//    rate = sqrt(750.0 * 500.0 / mask.cols / mask.rows);
//    cv::Size fit_size(mask.cols*rate, mask.rows*rate);
//
//    cv::Mat fit_obj1;
//    cv::resize(obj, fit_obj1, fit_size);
//    cv::Mat fit_obj2 = fit_obj1.clone();
//
//    for (int i = 0; i < fit_size.height; ++i)
//    {
//        for (int j = 0; j < fit_size.width; ++j)
//        {
//            if (middle_mat[0].at<uchar>(i, j) == 0)
//            {
//                fit_obj1.at<cv::Vec3b>(i, j)[0] = fit_obj1.at<cv::Vec3b>(i, j)[0] / 2 + 255 / 2;
//                fit_obj1.at<cv::Vec3b>(i, j)[1] = fit_obj1.at<cv::Vec3b>(i, j)[1] / 2;
//                fit_obj1.at<cv::Vec3b>(i, j)[2] = fit_obj1.at<cv::Vec3b>(i, j)[2] / 2;
//            }
//            if (middle_mat[0].at<uchar>(i, j) == 255)
//            {
//                fit_obj1.at<cv::Vec3b>(i, j)[0] = fit_obj1.at<cv::Vec3b>(i, j)[0] / 2;
//                fit_obj1.at<cv::Vec3b>(i, j)[1] = fit_obj1.at<cv::Vec3b>(i, j)[1] / 2;
//                fit_obj1.at<cv::Vec3b>(i, j)[2] = fit_obj1.at<cv::Vec3b>(i, j)[2] / 2 + 255 / 2;
//            }
//            if (middle_mat[1].at<uchar>(i, j) == 0)
//            {
//                fit_obj2.at<cv::Vec3b>(i, j)[0] *= 0.3;
//                fit_obj2.at<cv::Vec3b>(i, j)[1] *= 0.3;
//                fit_obj2.at<cv::Vec3b>(i, j)[2] *= 0.3;
//            }
//        }
//    }
//
//    //imshow("type - 2 - trip map", fit_obj1);
//    //imshow("type - 2 - mask previewer", fit_obj2);
//    imshow("抠图参数调整", fit_obj2);
//    return true;
//}
//
//bool mask_configurator::previewMask(std::unordered_map<std::string, int> &inputOutput_config)
//{
//    switch (inputOutput_config.at("make_type"))
//    {
//    case NO_PROCESS: return previewNoProcess(inputOutput_config);
//    case FRAME_DIFFERENCE: return previewFrameDifference(inputOutput_config);
//    case GLOBAL_MATTING: return previewGlobalMatting(inputOutput_config);
//    default: return false;
//    }
//}
//
//int mask_configurator::makeConfig(std::unordered_map<std::string, int> &inputOutput_config, const cv::Mat &preview_obj)
//{
//    checkConfig(inputOutput_config);
//    preview_obj.copyTo(obj);
//    switch (inputOutput_config.at("make_type"))
//    {
//    case NO_PROCESS:
//    {
//        destoryTypeWindows();
//        previewNoProcess(inputOutput_config);
//        int key_value = cv::waitKey(0);
//        destoryTypeWindows();
//        return key_value;
//    }
//    case FRAME_DIFFERENCE:
//    {
//        destoryTypeWindows();
//        //TODO:修改为FRAME_DIFFERENCE的方法，修改前暂定为NO_PROCESS方法
//        previewNoProcess(inputOutput_config);
//        int key_value = cv::waitKey(0);
//        destoryTypeWindows();
//        return key_value;
//    }
//    case GLOBAL_MATTING:
//    {
//        destoryTypeWindows();
//        int config_value[4];
//
//        config_value[0] = inputOutput_config.at("bg_threshold");
//        config_value[1] = inputOutput_config.at("fg_threshold");
//        config_value[2] = inputOutput_config.at("edge_smooth_kernal_size");
//        config_value[3] = inputOutput_config.at("mask_threshold");
//
//        //cv::namedWindow("type - 2 - trip map");
//        //cv::namedWindow("type - 2 - mask previewer");
//        //cv::createTrackbar("bg_threshold", "type - 2 - trip map",
//        //    &config_value[0], 255, on_Trackbar_config_value_1_0, &inputOutput_config);
//        //cv::createTrackbar("fg_threshold", "type - 2 - trip map",
//        //    &config_value[1], 30, on_Trackbar_config_value_1_1, &inputOutput_config);
//        //cv::createTrackbar("edge_smooth_kernal_size", "type - 2 - mask previewer",
//        //    &config_value[2], 30, on_Trackbar_config_value_1_2, &inputOutput_config);
//        //cv::createTrackbar("mask_threshold", "type - 2 - mask previewer",
//        //    &config_value[3], 100, on_Trackbar_config_value_1_3, &inputOutput_config);
//
//        if (config_value[0] > 50)
//        {
//            config_value[0] = 49.0 / 204.0*(config_value[0] - 255) + 100;
//            if (config_value[0] > 100) { config_value[0] = 100; }
//            else if (config_value[0] < 0) { config_value[0] = 0; }
//        }
//
//
//        cv::namedWindow("抠图参数调整");
//        cv::createTrackbar("背景阈值", "抠图参数调整",
//            &config_value[0], 100, on_Trackbar_config_value_1_0, &inputOutput_config);
//        cv::createTrackbar("前景阈值", "抠图参数调整",
//            &config_value[1], 30, on_Trackbar_config_value_1_1, &inputOutput_config);
//        cv::createTrackbar("边缘光滑", "抠图参数调整",
//            &config_value[2], 100, on_Trackbar_config_value_1_2, &inputOutput_config);
//        cv::createTrackbar("maks大小调整", "抠图参数调整",
//            &config_value[3], 100, on_Trackbar_config_value_1_3, &inputOutput_config);
//
//        if (!previewGlobalMatting(inputOutput_config))
//        {
//            destoryTypeWindows();
//            return 27;//"ESC"
//        }
//        int key_value = cv::waitKey(0);
//        destoryTypeWindows();
//        return key_value;
//    }
//    default:
//        return 27;//"ESC"
//    }
//}
//
//bool mask_configurator::saveConfig(const std::unordered_map<std::string, int> &config)
//{
//    std::ofstream file(config_file_save_path);
//    if (file.is_open())
//    {
//        //std::cout << "save "<< std::endl;
//        //std::cout << config_file_save_path << std::endl;
//        for (auto itr = config.begin(); itr != config.end(); ++itr)
//        {
//            if (itr->first == "make_type")
//            {
//                file << itr->first << "=";
//                switch (itr->second)
//                {
//                case GLOBAL_MATTING:file << "GLOBAL_MATTING"; break;
//                default:file << "GLOBAL_MATTING"; break;
//                }
//                file << std::endl;
//            }
//            else
//            {
//                file << itr->first << "=" << itr->second << std::endl;
//            }
//        }
//        return true;
//    }
//    else return false;
//}
//
//bool mask_configurator::saveConfig(const std::unordered_map<std::string, int> &config,const std::string &_config_file_save_path)
//{
//    std::ofstream file(_config_file_save_path);
//    if (file.is_open())
//    {
//        //std::cout << "save "<< std::endl;
//        //std::cout << config_file_save_path << std::endl;
//        for (auto itr = config.begin(); itr != config.end(); ++itr)
//        {
//            if (itr->first == "make_type")
//            {
//                file << itr->first << "=";
//                switch (itr->second)
//                {
//                case GLOBAL_MATTING:file << "GLOBAL_MATTING"; break;
//                default:file << "GLOBAL_MATTING"; break;
//                }
//                file << std::endl;
//            }
//            else
//            {
//                file << itr->first << "=" << itr->second << std::endl;
//            }
//        }
//        return true;
//    }
//    else return false;
//}
//
//void mask_configurator::printConfig(const std::unordered_map<std::string, int> &config)
//{
//    for (auto itr = config.begin(); itr != config.end(); ++itr)
//    {
//        if (itr->first == "make_type")
//        {
//            std::cout << itr->first << "=";
//            switch (itr->second)
//            {
//            case GLOBAL_MATTING:std::cout << "GLOBAL_MATTING"; break;
//            default:std::cout << "GLOBAL_MATTING";break;
//            }
//            std::cout << std::endl;
//        }
//        else
//        {
//            std::cout << itr->first << "=" << itr->second << std::endl;
//        }
//    }
//    return;
//}
//
//bool mask_configurator::isSameConfig(const mask_configuration &config1, const mask_configuration &config2)
//{
//    if (config1.make_type != config2.make_type) return false;
//    if (config1.key.size() != config2.key.size()) return false;
//    if (config1.configuration.size() != config2.configuration.size()) return false;
//    if (config1.key.size() != config1.configuration.size()) return false;
//    for (int i = 0; i < config1.key.size(); ++i)
//    {
//        if(config1.key[i] != config2.key[i]) return false;
//        if (config1.configuration[i] != config2.configuration[i]) return false;
//    }
//    return true;
//}
//
//void mask_configurator::configCopyTo(const mask_configuration &config1, mask_configuration &config2)
//{
//    config2.make_type = config1.make_type;
//    config2.key.assign(config1.key.begin(), config1.key.end());
//    config2.configuration.assign(config1.configuration.begin(), config1.configuration.end());
//}
//
//void mask_configurator::on_Trackbar_config_value_0(int bar_value, void *v_p_config)
//{
//    mask_configuration * p_config = (mask_configuration *)v_p_config;
//    p_config->configuration[0] = bar_value;
//    previewMask(*p_config);
//    return;
//}
//
//void mask_configurator::on_Trackbar_config_value_1(int bar_value, void *v_p_config)
//{
//    mask_configuration * p_config = (mask_configuration *)v_p_config;
//    p_config->configuration[1] = bar_value;
//    previewMask(*p_config);
//    return;
//}
//
//void mask_configurator::on_Trackbar_config_value_2(int bar_value, void *v_p_config)
//{
//    mask_configuration * p_config = (mask_configuration *)v_p_config;
//    p_config->configuration[2] = bar_value;
//    previewMask(*p_config);
//    return;
//}
//
//void mask_configurator::on_Trackbar_config_value_3(int bar_value, void *v_p_config)
//{
//    mask_configuration * p_config = (mask_configuration *)v_p_config;
//    p_config->configuration[3] = bar_value;
//    previewMask(*p_config);
//    return;
//}
//
//void mask_configurator::on_Trackbar_config_value_others(int bar_value, void *v_p_config)
//{
//    return;
//}
//
//bool mask_configurator::checkConfig(mask_configuration &config)
//{
//    bool flag = false;
//    switch (config.make_type)
//    {
//    case NO_PROCESS:
//    {
//        config.key.resize(0);
//        config.configuration.resize(0);
//        flag = true;
//        break;
//    }
//    case FRAME_DIFFERENCE:
//    {
//        config.key.resize(0);
//        config.configuration.resize(0);
//        flag = true;
//        break;
//    }
//    case GLOBAL_MATTING:
//    {
//        if (config.configuration.size() >= 4)
//            if (0 <= config.configuration[0] && config.configuration[0] <= 255)
//                if (0 <= config.configuration[1] && config.configuration[1] <= 30)
//                    if (0 <= config.configuration[2] && config.configuration[2] <= 30)
//                        if (0 <= config.configuration[3] && config.configuration[3] <= 100)
//                            flag = true;
//        if (!flag)
//        {
//            config.make_type = GLOBAL_MATTING;
//            config.key.resize(4);
//            config.configuration.resize(4);
//            config.key[0] = "bg_threshold";
//            config.configuration[0] = 30;
//            config.key[1] = "fg_threshold";
//            config.configuration[1] = 3;
//            config.key[2] = "edge_smooth_kernal_size";
//            config.configuration[2] = 7;
//            config.key[3] = "mask_threshold";
//            config.configuration[3] = 50;
//
//        }
//        break;
//    }
//    default:
//    {
//        config.make_type = GLOBAL_MATTING;
//        config.key.resize(4);
//        config.configuration.resize(4);
//        config.key[0] = "bg_threshold";
//        config.configuration[0] = 30;
//        config.key[1] = "fg_threshold";
//        config.configuration[1] = 3;
//        config.key[2] = "edge_smooth_kernal_size";
//        config.configuration[2] = 7;
//        config.key[3] = "mask_threshold";
//        config.configuration[3] = 50;
//        break;
//    }
//    }
//    return flag;
//}
//
//bool mask_configurator::previewNoProcess(mask_configuration &inputOutput_config)
//{
//    cv::Mat mask = mask_maker::createMask(obj, inputOutput_config);
//    if (mask.empty()) return false;//"ESC"
//
//    double rate = 1;
//    rate = sqrt(750.0 * 500.0 / mask.cols / mask.rows);
//    cv::Size fit_size(mask.cols*rate, mask.rows*rate);
//    cv::Mat fit_obj;    
//    cv::resize(obj, fit_obj, fit_size);
//
//    for (int i = 0; i < fit_obj.rows; ++i)
//    {
//        for (int j = 0; j < fit_obj.cols; ++j)
//        {
//            if (mask.at<uchar>(i, j) == 0)
//            {
//                fit_obj.at<cv::Vec3b>(i, j)[0] *= 0.3;
//                fit_obj.at<cv::Vec3b>(i, j)[1] *= 0.3;
//                fit_obj.at<cv::Vec3b>(i, j)[2] *= 0.3;
//            }
//        }
//    }
//    imshow("type - 0 - mask previewer", fit_obj);
//    return true;
//}
//
//bool mask_configurator::previewFrameDifference(mask_configuration &inputOutput_config)
//{
//    return false;//"ESC"
//}
//
//bool mask_configurator::previewGlobalMatting(mask_configuration &inputOutput_config)
//{
//    std::vector<cv::Mat> middle_mat;
//    cv::Mat mask = mask_maker::createMask(obj, inputOutput_config, middle_mat);
//    if (mask.empty()) return false;//"ESC"
//
//    double rate = 1;
//    rate = sqrt(750.0 * 500.0 / mask.cols / mask.rows);
//    cv::Size fit_size(mask.cols*rate, mask.rows*rate);
//
//    cv::Mat fit_obj1;
//    cv::resize(obj, fit_obj1, fit_size);
//    cv::Mat fit_obj2 = fit_obj1.clone();
//
//    for (int i = 0; i < fit_size.height; ++i)
//    {
//        for (int j = 0; j < fit_size.width; ++j)
//        {
//            if (middle_mat[0].at<uchar>(i, j) == 0)
//            {
//                fit_obj1.at<cv::Vec3b>(i, j)[0] = fit_obj1.at<cv::Vec3b>(i, j)[0] / 2 + 255 / 2;
//                fit_obj1.at<cv::Vec3b>(i, j)[1] = fit_obj1.at<cv::Vec3b>(i, j)[1] / 2;
//                fit_obj1.at<cv::Vec3b>(i, j)[2] = fit_obj1.at<cv::Vec3b>(i, j)[2] / 2;
//            }
//            if (middle_mat[0].at<uchar>(i, j) == 255)
//            {
//                fit_obj1.at<cv::Vec3b>(i, j)[0] = fit_obj1.at<cv::Vec3b>(i, j)[0] / 2;
//                fit_obj1.at<cv::Vec3b>(i, j)[1] = fit_obj1.at<cv::Vec3b>(i, j)[1] / 2;
//                fit_obj1.at<cv::Vec3b>(i, j)[2] = fit_obj1.at<cv::Vec3b>(i, j)[2] / 2 + 255 / 2;
//            }
//            if (middle_mat[1].at<uchar>(i, j) == 0)
//            {
//                fit_obj2.at<cv::Vec3b>(i, j)[0] *= 0.3;
//                fit_obj2.at<cv::Vec3b>(i, j)[1] *= 0.3;
//                fit_obj2.at<cv::Vec3b>(i, j)[2] *= 0.3;
//            }
//        }
//    }
//
//    imshow("type - 2 - trip map", fit_obj1);
//    imshow("type - 2 - mask previewer", fit_obj2);
//    return true;
//}
//
//
//bool mask_configurator::previewMask(mask_configuration &inputOutput_config)
//{
//    switch (inputOutput_config.make_type)
//    {
//    case NO_PROCESS: return previewNoProcess(inputOutput_config);
//    case FRAME_DIFFERENCE: return previewFrameDifference(inputOutput_config);
//    case GLOBAL_MATTING: return previewGlobalMatting(inputOutput_config);
//    default: return false;
//    }
//}
//
//void mask_configurator::destoryTypeWindows()
//{
//    //NO_PROCESS: 
//
//    //FRAME_DIFFERENCE: 
//
//    //GLOBAL_MATTING:
//    cv::destroyWindow("type - 2 - trip map");
//    cv::destroyWindow("type - 2 - mask previewer");
//
//    return;
//
//}
//
//
//int mask_configurator::makeConfig(mask_configuration &inputOutput_config, const cv::Mat &preview_obj)
//{
//    checkConfig(inputOutput_config);
//    preview_obj.copyTo(obj);
//    switch (inputOutput_config.make_type)
//    {
//    case NO_PROCESS: 
//    {
//        destoryTypeWindows();
//        previewNoProcess(inputOutput_config);
//        int key_value = cv::waitKey(0);
//        destoryTypeWindows();
//        return key_value;
//    }
//    case FRAME_DIFFERENCE: 
//    {
//        destoryTypeWindows();
//        //TODO:修改为FRAME_DIFFERENCE的方法，修改前暂定为NO_PROCESS方法
//        previewNoProcess(inputOutput_config);
//        int key_value = cv::waitKey(0);
//        destoryTypeWindows();
//        return key_value;
//    }
//    case GLOBAL_MATTING: 
//    {
//        destoryTypeWindows();
//        int config_value[4];
//        for (int i = 0; i < inputOutput_config.configuration.size(); ++i)
//        {
//            config_value[i] = inputOutput_config.configuration[i];
//        }
//        cv::namedWindow("type - 2 - trip map");
//        cv::namedWindow("type - 2 - mask previewer");
//        cv::createTrackbar(inputOutput_config.key[0], "type - 2 - trip map",
//            &config_value[0], 255, on_Trackbar_config_value_0, &inputOutput_config);
//        cv::createTrackbar(inputOutput_config.key[1], "type - 2 - trip map",
//            &config_value[1], 30, on_Trackbar_config_value_1, &inputOutput_config);
//        cv::createTrackbar(inputOutput_config.key[2], "type - 2 - mask previewer",
//            &config_value[2], 30, on_Trackbar_config_value_2, &inputOutput_config);
//        cv::createTrackbar(inputOutput_config.key[3], "type - 2 - mask previewer",
//            &config_value[3], 100, on_Trackbar_config_value_3, &inputOutput_config);
//        if (!previewGlobalMatting(inputOutput_config))
//        {
//            destoryTypeWindows();
//            return 27;//"ESC"
//        }
//        int key_value = cv::waitKey(0);
//        destoryTypeWindows();
//        return key_value;
//    }
//    default:
//        return 27;//"ESC"
//    }
//}
//
//
//
//bool mask_configurator::saveConfig(const mask_configuration &inputOutput_config)
//{
//    std::ofstream file(config_file_save_path);
//    
//    if(file.is_open())
//    {
//        switch (inputOutput_config.make_type)
//        {
//        case NO_PROCESS:
//            file << "NO_PROCESS" << std::endl; 
//            break;
//        case FRAME_DIFFERENCE:
//            file << "FRAME_DIFFERENCE" << std::endl; 
//            break;
//        case GLOBAL_MATTING:
//            file << "GLOBAL_MATTING" << std::endl;
//            break;
//        }
//        for (int i = 0; i < inputOutput_config.key.size()
//            && i < inputOutput_config.configuration.size(); ++i)
//        {
//            std::string key = inputOutput_config.key[i].substr(0,
//                inputOutput_config.key[i].find_first_of(' '));
//            if (key == "")
//                file << "UNKNOWN" << "    ";
//            else
//                file << key << "    ";
//            file << inputOutput_config.configuration[i] << std::endl;
//        }
//        return true;
//    }
//    else return false;
//}
//
//void mask_configurator::printConfig(const mask_configuration &inputOutput_config)
//{
//    switch (inputOutput_config.make_type)
//    {
//    case NO_PROCESS:
//        std::cout << "NO_PROCESS" << std::endl;
//        break;
//    case FRAME_DIFFERENCE:
//        std::cout << "FRAME_DIFFERENCE" << std::endl;
//        break;
//    case GLOBAL_MATTING:
//        std::cout << "GLOBAL_MATTING" << std::endl;
//        break;
//    }
//    for (int i = 0; i < inputOutput_config.key.size()
//        && i < inputOutput_config.configuration.size(); ++i)
//    {
//        std::string key = inputOutput_config.key[i].substr(0,
//            inputOutput_config.key[i].find_first_of(' '));
//        if (key == "")
//            std::cout << "UNKNOWN" << "    ";
//        else
//            std::cout << key << "    ";
//        std::cout << inputOutput_config.configuration[i] << std::endl;
//    }
//    return;
//}
//
//mask_configuration mask_configurator::getConfig()
//{
//    mask_configuration r;
//    r.make_type = -1;
//    std::ifstream file(config_file_save_path);
//    if (file.is_open())
//    {
//        std::string type;
//        file >> type;
//        if (type == "NO_PROCESS") r.make_type = NO_PROCESS;
//        else if (type == "FRAME_DIFFERENCE") r.make_type = FRAME_DIFFERENCE;
//        else if (type == "GLOBAL_MATTING") r.make_type = GLOBAL_MATTING;
//        else r.make_type = -1;
//        while (!file.eof())
//        {
//            std::string key;
//            int value;
//            file >> key;
//            file >> value;
//            r.configuration.push_back(value);
//            r.key.push_back(key);
//        }
//    }
//    return r;
//}
//
//
//bool mask_configurator::save(const std::vector<std::string> &obj_path, const std::vector<std::string> &label, 
//    const std::string &save_path,const std::string &config_name, const std::unordered_map<std::string, int> &config)
//{
//    _mkdir((save_path).c_str());
//    std::string step_0_save_path = save_path + "\\step0";
//    _mkdir(step_0_save_path.c_str());
//    std::ofstream file(step_0_save_path + "\\list.txt");
//    std::string obj_sava_path = step_0_save_path + "\\obj";
//    std::string mask_sava_path = step_0_save_path + "\\mask";
//
//    _mkdir((obj_sava_path).c_str());
//    _mkdir((mask_sava_path).c_str());
//
//    if (!file.is_open()) return false;
//
//    for (int i = 0; i < obj_path.size(); ++i)
//    {
//        cv::Mat obj = cv::imread(obj_path[i]);
//        if (obj.empty()) continue;
//        cv::Mat mask = mask_maker::createMask(obj, config);
//        std::string front_name;
//        std::string extend_name;
//        std::string obj_name;
//        std::expand::get_path_last_name(obj_path[i], obj_name, front_name, extend_name);
//        //std::string maske_name = front_name + "_mask." + extend_name;
//        std::string maske_name = front_name + "_mask." + "png";
//        cv::imwrite(obj_sava_path + "\\" + obj_name, obj);//obj
//        cv::imwrite(mask_sava_path + "\\" + maske_name, mask);//mask
//        file << obj_sava_path << "\\" << obj_name;
//        file << " ";
//        file << mask_sava_path << "\\" << maske_name;
//        file << " ";
//        file << label[i];
//        file << std::endl;
//    }
//    return saveConfig(config, save_path + "\\" + config_name);
//}
