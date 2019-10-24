#include"poisson_blender.h"
#include"stl_expand/stl_expand.hpp"

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<vector>
#include<fstream>

#include<random>
//test
#include<iostream>
#include<iomanip>


//using namespace std;

poisson_blender::poisson_blender(double size_rate,int itr,double th)
{
	setParam(size_rate, itr, th);
	//srand((unsigned)time(NULL));
	initK();
    initRand();
}

poisson_blender::~poisson_blender() {}

void poisson_blender::setParam(double size_rate, int itr, double th)
{
	_size_rate = size_rate;
	_itr = itr;
	_th = th;
	return;
}

void poisson_blender::initK()
{
	cv::Mat(3, 3, CV_32FC1).copyTo(K);
	K.at<float>(0, 0) = 0.0f; K.at<float>(0, 1) = 1.0f; K.at<float>(0, 2) = 0.0f;
	K.at<float>(1, 0) = 1.0f; K.at<float>(1, 1) = 0.0f; K.at<float>(1, 2) = 1.0f;
	K.at<float>(2, 0) = 0.0f; K.at<float>(2, 1) = 1.0f; K.at<float>(2, 2) = 0.0f;

	cv::Mat(1, 3, CV_32FC1).copyTo(Kh);
	Kh.at<float>(0, 0) = 0.0f; Kh.at<float>(0, 1) = -1.0f; Kh.at<float>(0, 2) = 1.0f;

	cv::Mat(3, 1, CV_32FC1).copyTo(Kv);
	Kv.at<float>(0, 0) = 0.0f;
	Kv.at<float>(1, 0) = -1.0f;
	Kv.at<float>(2, 0) = 1.0f;

	return;
}


void poisson_blender::initRand()
{
    rand_engine = std::default_random_engine(time(0));
    rand_dist = std::uniform_int_distribution<size_t>(0, 32768-1);
    max_rand = 32768-1;
}

size_t poisson_blender::get_rand()
{
    return rand_dist(rand_engine);
}

void poisson_blender::split_png(const cv::Mat &bgra, cv::Mat &bgr, cv::Mat &a)
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

cv::Rect poisson_blender::findMaskBoundingBox(const cv::Mat &bin_mask_8UC1)
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
        x2[i] = x1[i] + one_rect.width;
        y2[i] = y1[i] + one_rect.height;
    }

    cv::Rect output;
    output.x = *(std::min_element(x1.begin(), x1.end()));
    output.y = *(std::min_element(y1.begin(), y1.end()));
    output.width = *(std::max_element(x2.begin(), x2.end())) - output.x;
    output.height = *(std::max_element(y2.begin(), y2.end())) - output.y;

    return output;
}

cv::Size poisson_blender::get_fit_size(const cv::Size &bg_size, const cv::Size &mask_compact_size)
{
    double rate = sqrt((double(get_rand() % 1024) / 2048 + 0.5)*_size_rate * double(bg_size.area()) / double(mask_compact_size.area()));
    cv::Size _size(int(mask_compact_size.width *rate), int(mask_compact_size.height * rate));
    while (_size.width >= bg_size.width || _size.height >= bg_size.height)
    {
        _size.width *= 0.8;
        _size.height *= 0.8;
    }
    return _size;
}

//cv::Rect poisson_blender::choicePlace(const cv::Mat &bg, const cv::Mat &mask)
//{
//	int d_width = bg.cols - mask.cols;
//	int d_height = bg.rows - mask.rows;
//	cv::Rect r(rand() % d_width, rand() % d_height, mask.cols, mask.rows);
//	return r;
//}

bool poisson_blender::isIntersect(const cv::Rect &region1, const cv::Rect &region2)
{
	int x11 = region1.x;
	int y11 = region1.y;
	int x12 = region1.x + region1.width;
	int y12 = region1.y + region1.height;
	int x21 = region2.x;
	int y21 = region2.y;
	int x22 = region2.x + region2.width;
	int y22 = region2.y + region2.height;

	//std::cout << x11 << "  " << y11 << "  " << x12 << "  " << y12 << "  " << x21 << "  " << y21 << "  " << x22 << "  " << y22 << "    ";

	if (x12 < x21) return false;
	if (x22 < x11) return false;
	if (y12 < y21) return false;
	if (y22 < y11) return false;

	return true;

	//int minX = x11 > x21 ? x11 : x21;//max(x11,x21)
	//int minY = y11 > y21 ? y11 : y21;//max(y11,y21)
	//int maxX = x12 < x22 ? x12 : x22;//min(x12,x22)
	//int maxY = y12 < y22 ? y12 : y22;//min(y12,y22)

	//if (minX < maxX && minY < maxY) return true;
	//else return false;
}

bool poisson_blender::choicePlace(const cv::Size &bg_size, const cv::Size &mask_size, std::vector<cv::Rect> &reject_region, cv::Rect &output)
{
    clock_t begin, end;
    int i_for_while = 0;

	int d_width = bg_size.width - mask_size.width;
	int d_height = bg_size.height - mask_size.height;
	output.width = mask_size.width;
    output.height = mask_size.height;

	bool find_OK_rect = false;

    while (!find_OK_rect)
    {
        output.x = get_rand() % d_width;
        output.y = get_rand() % d_height;
        find_OK_rect = true;
        for (int j = 0; j < reject_region.size(); ++j)
        {
            if (isIntersect(output, reject_region[j]))
            {
                find_OK_rect = false;
                break;
            }
        }
        if (i_for_while == 100)
        {
            end = clock();
            if (double(end - begin) / CLOCKS_PER_SEC > 60.0) { break; }
            i_for_while = -1;
        }
        i_for_while++;
    }
    if (!find_OK_rect)
    {
        return false;
    }
	reject_region.push_back(output);
	return true;
}

double poisson_blender::coverRate(const cv::Rect &region1, const cv::Rect &region2)
{
    if (!isIntersect(region1, region2)) { return 0; }
    cv::Point tl = region2.tl();
    cv::Point tr(region2.x + region2.width, region2.y);
    cv::Point br = region2.br();
    cv::Point bl(region2.x, region2.y + region2.height);
    bool is_contain_tl = region1.contains(tl);
    bool is_contain_tr = region1.contains(tr);
    bool is_contain_br = region1.contains(br);
    bool is_contain_bl = region1.contains(bl);

    double cover_width, cover_height;
    //0000
    if (!is_contain_tl && !is_contain_tr && !is_contain_br && !is_contain_bl)
    {
        if (region1.x < region2.x&&region2.x < region1.x + region1.width)
        {
            if (region1.x + region1.width < region2.x + region2.width)
            {
                cover_width = region1.x + region1.width - region2.x;
                cover_height = region1.height;
            }
            else
            {
                cover_width = region2.width;
                cover_height = region1.height;
            }
        }
        else if (region1.y < region2.y&&region2.y < region1.y + region1.height)
        {
            if (region1.y + region1.height < region2.y + region2.height)
            {
                cover_width = region1.width;
                cover_height = region1.y + region1.height - region2.y;
            }
            else
            {
                cover_width = region1.width;
                cover_height = region2.height;
            }
        }
        else
        {
            return 1.0;
        }
    }
    else if(is_contain_tl&&is_contain_br|| is_contain_tr&&is_contain_bl)
    {
        cover_width = region2.width;
        cover_height = region2.height;
    }
    //0011 
    else if (!is_contain_tl && !is_contain_tr && is_contain_br && is_contain_bl)
    {
        cover_width = region2.width; 
        cover_height = region2.y + region2.height - region1.x;
    }
    //1100
    else if (is_contain_tl && is_contain_tr && !is_contain_br && !is_contain_bl)
    {
        cover_width = region2.width;
        cover_height = region1.y + region1.height - region2.x;
    }
    //0110
    else if (!is_contain_tl && is_contain_tr && is_contain_br && !is_contain_bl)
    {
        cover_width = region2.x + region2.width - region1.x;
        cover_height = region2.height;
    }
    //1001
    else if (is_contain_tl && !is_contain_tr && !is_contain_br && is_contain_bl)
    {
        cover_width = region1.x + region1.width - region2.x;
        cover_height = region2.height;
    }
    //0010
    else if (!is_contain_tl && !is_contain_tr && is_contain_br && !is_contain_bl)
    {
        cover_width = (region2.x + region2.width - region1.x); 
        cover_height = (region2.y + region2.height - region1.y);
    }
    //0001
    else if (!is_contain_tl && !is_contain_tr && !is_contain_br && is_contain_bl)
    {
        cover_width = (region1.x + region1.width - region2.x);
        cover_height = (region2.y + region2.height - region1.y);
    }
    //1000
    else if (is_contain_tl && !is_contain_tr && !is_contain_br && !is_contain_bl)
    {
        cover_width = (region1.x + region1.width - region2.x);
        cover_height = (region1.y + region1.height - region2.y);
    }
    //0100
    else if (is_contain_tl && !is_contain_tr && !is_contain_br && !is_contain_bl)
    {
        cover_width = (region2.x + region2.width - region1.x);
        cover_height = (region1.y + region1.height - region2.y);
    }
    return cover_width*cover_height / region1.area();
}

bool poisson_blender::choicePlace2Cover(const cv::Size &bg_size, const cv::Size &mask_size, std::vector<cv::Rect> &reject_region, cv::Rect &output, double min_cover_rate, double max_cover_rate)
{
    clock_t begin, end;
    int i_for_while = 0;
    int d_width = bg_size.width - mask_size.width;
    int d_height = bg_size.height - mask_size.height;
    output.width = mask_size.width;
    output.height = mask_size.height;

    int cover_case = reject_region.size() % 2;
    bool find_OK_rect = false;
    switch (cover_case)
    {
    case 0:
    {
        //randon no cover place
        begin = clock();
        i_for_while = 0;
        while (!find_OK_rect)
        {
            output.x = get_rand() % d_width;
            output.y = get_rand() % d_height;
            find_OK_rect = true;
            for (int j = 0; j < reject_region.size(); ++j)
            {
                if (isIntersect(output, reject_region[j]))
                {
                    find_OK_rect = false;
                    break;
                }
            }
            if (i_for_while == 100)
            {
                end = clock();
                if (double(end - begin) / CLOCKS_PER_SEC > 60.0) { break; }
                i_for_while = -1;
            }
            i_for_while++;
        }
        break;
    }
    case 1:
    {

        cv::Rect &last_region = reject_region[reject_region.size() - 1];
        int x_min = last_region.x - output.width;
        if (x_min < 0) { x_min = 0; }
        int x_max = last_region.x + last_region.width - 1;
        if (x_max > d_width - 1) { x_max = d_width - 1; }
        int y_min = last_region.y - output.height;
        if (y_min < 0) { y_min = 0; }
        int y_max = last_region.y + last_region.height - 1;
        if (y_max > d_height - 1) { y_max = d_height - 1; }

        begin = clock();
        i_for_while = 0;
        //random some cover place
        while (!find_OK_rect)
        {
            output.x = get_rand() % d_width;
            output.y = get_rand() % d_height;
            int num_exist_accept = 0;
            find_OK_rect = true;
            for (int j = 0; j < reject_region.size(); ++j)
            {
                double _coverRate = coverRate(reject_region[j], output);
                if (min_cover_rate <= _coverRate && _coverRate <= max_cover_rate)
                {
                    num_exist_accept++;
                }
                else if (_coverRate > max_cover_rate)
                {
                    find_OK_rect = false;
                    break;
                }
            }
            //rechoice
            if (find_OK_rect&&num_exist_accept != 0) { break; }

            output.x = double(get_rand()) / RAND_MAX * (x_max - x_min) + x_min;
            output.y = double(get_rand()) / RAND_MAX * (y_max - y_min) + y_min;
            find_OK_rect = true;
            double _last_coverRate = coverRate(reject_region[reject_region.size() - 1], output);
            if (min_cover_rate <= _last_coverRate && _last_coverRate <= max_cover_rate)
            {
                int j = 0;
                for (; j < reject_region.size() - 1; ++j)
                {
                    if (coverRate(reject_region[j], output)> max_cover_rate)
                    {
                        find_OK_rect = false;
                        break;
                    }
                }
                if (j == reject_region.size() - 1)
                {
                    find_OK_rect = true;
                }
            }
            else
            {
                find_OK_rect = false;
            }
            if (i_for_while == 100)
            {
                end = clock();
                if (double(end - begin) / CLOCKS_PER_SEC > 60.0) { break; }
                i_for_while = -1;
            }
            i_for_while++;
        }
        break;
    }
    }
    if (!find_OK_rect)
    {
        return false;
    }
    reject_region.push_back(output);
    return true;
}

//cv::Rect poisson_blender::findObjectBoundingBox(const cv::Mat &Mask)
//{
//    return mask_maker::findObjectBoundingBox(Mask);
//
//	//std::vector<std::vector<cv::Point>> contours;
//	//std::vector<cv::Vec4i> hierarchy;
//
//	////cv::Mat _mask;
//	////cv::cvtColor(Mask, _mask, CV_BGR2GRAY);
//	////findContours(_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
//
//
//	//findContours(Mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
//
//
//	//std::vector<std::vector<cv::Point> > contours_poly(contours.size());
//	//std::vector<cv::Rect> boundRect(contours.size());
//	//for (int i = 0; i < contours.size(); i++)
//	//{
//	//	cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
//	//	boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
//	//}
//	//int index = 0;
//	//double area = -1;
//	//for (int i = 0; i < boundRect.size(); ++i)
//	//{
//	//	double t_area = boundRect[i].area();
//	//	if (t_area > area)
//	//	{
//	//		index = i;
//	//		area = t_area;
//	//	}
//	//}
//	//return boundRect[index];
//}

void poisson_blender::imgrad(const cv::Mat &bgr_image, cv::Mat &Fh_out, cv::Mat &Fv_out)
{
	//cout << CV_8UC1 << endl;
	//cout << bgr_image.type() << "    " << Fh_out.type() << "    " << Fv_out.type() << endl;
	cv::filter2D(bgr_image, Fh_out, CV_32F, Kh);
	//for (int i = 0; i < Fh_out.rows; ++i)
	//{
	//	Fh_out.at<cv::Vec3f>(i, Fh_out.cols - 1)[0] = 0;
	//	Fh_out.at<cv::Vec3f>(i, Fh_out.cols - 1)[1] = 0;
	//	Fh_out.at<cv::Vec3f>(i, Fh_out.cols - 1)[2] = 0;
	//}
	cv::filter2D(bgr_image, Fv_out, CV_32F, Kv);
	//for (int j = 0; j < Fv_out.cols; ++j)
	//{
	//	Fv_out.at<cv::Vec3f>(Fv_out.rows - 1, j)[0] = 0;
	//	Fv_out.at<cv::Vec3f>(Fv_out.rows - 1, j)[1] = 0;
	//	Fv_out.at<cv::Vec3f>(Fv_out.rows - 1, j)[2] = 0;
	//}
	return;
}

void poisson_blender::PoissonJacobiSam(cv::Mat &mix_image, const cv::Mat &X, const cv::Mat &Y, const cv::Mat &Mask, int itr, double th)
{
    //clock_t begin, end;

    //begin = clock();
	//lap
	cv::Mat lap(X.rows, X.cols, CV_32FC3);

    float *p_lap = lap.ptr<float>();
 //   const float *p_X = X.ptr<float>();
 //   const float *p_Y = Y.ptr<float>();
 //   int _rows = X.rows, _cols = X.cols;
 //   for (int i = 0; i < lap.cols; ++i)
 //   {
 //       for (int j = 0; j < lap.rows; ++j)
 //       {
 //           //lap.at<cv::Vec3f>(i, j)
 //           //	= X.at<cv::Vec3f>(i, (j - 1 + X.cols) % X.cols)
 //           //	+ Y.at<cv::Vec3f>((i - 1 + Y.rows) % Y.rows, j)
 //           //	- X.at<cv::Vec3f>(i, j)
 //           //	- Y.at<cv::Vec3f>(i, j);
 //           for (int k = 0; k < 3; ++k)
 //           {
 //               p_lap[(i*_cols + j) * 3 + k]
 //                   = p_X[(i*_cols + (j - 1 + _cols) % _cols) * 3 + k]
 //                   + p_Y[(((i - 1 + _rows) % _rows)*_cols + j) * 3 + k]
 //                   - p_X[(i*_cols + j) * 3 + k]
 //                   - p_Y[(i*_cols + j) * 3 + k];
 //           }
 //       }
 //   }

	for (int i = 0; i < lap.rows;i++)
	{
        
        cv::Vec3f * lap_i_line = lap.ptr<cv::Vec3f>(i);
        const cv::Vec3f * X_i_line = X.ptr<cv::Vec3f>(i);
        const cv::Vec3f * Y_i_line = Y.ptr<cv::Vec3f>(i);

        const cv::Vec3f * Y_one_line = Y.ptr<cv::Vec3f>((i - 1 + Y.rows) % Y.rows);

		for (int j = 0; j < lap.cols; j++)
		{
            lap_i_line[j]
                = X_i_line[(j - 1 + X.cols) % X.cols]
                + Y_one_line[j]
                - X_i_line[j]
                - Y_i_line[j];
		}
	}

	cv::Mat pre_mix_image;
	cv::Mat lpf;
	cv::Mat t_dst;
	cv::Mat t_dif;
	double max_dif;
	double min_dif;
	double dif0 = 1e30;
	
	cv::Mat mix_image_32FC3;
	mix_image.convertTo(mix_image_32FC3, CV_32FC3);

    //int t_itr = itr;
    //end = clock();
    //std::cout << double(end - begin) / CLOCKS_PER_SEC * 1000 << "    ";
    //begin = clock();

    cv::Mat clone_mix_image_32FC3[2];
    clone_mix_image_32FC3[0] = mix_image_32FC3.clone();
    clone_mix_image_32FC3[1] = mix_image_32FC3.clone();
    int i_clone = 0;
    int i_pre_clone = 1;

    std::vector<cv::Mat> v_Mask_8UC3(3);
    v_Mask_8UC3[0] = Mask;
    v_Mask_8UC3[1] = Mask;
    v_Mask_8UC3[2] = Mask;
    cv::Mat Mask_8UC3;
    cv::merge(v_Mask_8UC3, Mask_8UC3);
    const uchar *p_Mask_8UC3 = Mask_8UC3.ptr<uchar>();

	while (itr > 0)
	//if(1)
	{
        itr--;
        cv::filter2D(clone_mix_image_32FC3[i_pre_clone], clone_mix_image_32FC3[i_clone], CV_32F, K);
        float *p_clone_mix_image_32FC3 = clone_mix_image_32FC3[i_clone].ptr<float>();
        float *p_pre_clone_mix_image_32FC3 = clone_mix_image_32FC3[i_pre_clone].ptr<float>();

        int n = clone_mix_image_32FC3[i_clone].rows*clone_mix_image_32FC3[i_clone].cols * 3;
        max_dif = 0;
        for (int i = 0; i < n; ++i)
        {
            if (p_Mask_8UC3[i] == 0)
            {
                p_clone_mix_image_32FC3[i] = p_pre_clone_mix_image_32FC3[i];
            }
            else
            {
                //p_clone_mix_image_32FC3[i] /= 2.0f;
                p_clone_mix_image_32FC3[i] = (p_clone_mix_image_32FC3[i] + p_lap[i])*0.25f;
                float _abs_value = p_pre_clone_mix_image_32FC3[i] - p_clone_mix_image_32FC3[i];
                if (_abs_value > 0)
                {
                    if (_abs_value > max_dif)
                    {
                        max_dif = _abs_value;
                    }
                }
                else 
                {
                    if (-_abs_value > max_dif)
                    {
                        max_dif = -_abs_value;
                    }
                }

            }
        }

        if (i_clone == 0) { i_clone = 1; i_pre_clone = 0; }
        else { i_clone = 0; i_pre_clone = 1; }

        if ((dif0 - max_dif) / dif0 < th || max_dif < 0.25) { break; }
        //if ((dif0 - max_dif) / dif0 < th || max_dif < 1) { break; }
        dif0 = max_dif;

		//mix_image_32FC3.copyTo(pre_mix_image);
		//itr--;

		//cv::filter2D(mix_image_32FC3, lpf, CV_32F, K);
		//t_dst = ((lap + lpf) / 4.0f);
		//t_dst.copyTo(mix_image_32FC3, Mask);
		//cv::absdiff(mix_image_32FC3, pre_mix_image, t_dif);
		//cv::minMaxLoc(t_dif, &min_dif, &max_dif);
  //      if ((dif0 - max_dif) / dif0 < th || max_dif < 0.25) { break; }
		//dif0 = max_dif;
	}
    //mix_image_32FC3.convertTo(mix_image, CV_8UC3);

    clone_mix_image_32FC3[i_pre_clone].convertTo(mix_image, CV_8UC3);


    //end = clock();
    //std::cout <<  t_itr - itr << "  ";
    //std::cout << double(end - begin) / CLOCKS_PER_SEC * 1000 << "    ";

	return;
}

bool poisson_blender::save(const cv::Mat &image, 
	const std::string &filename, const std::string &image_path, const std::string &xml_path,
	std::vector<cv::Rect> &bndbox, const std::vector<std::string> &label)
{
	//static long num = 1000;
	//std::string xml_name = "fusion";
	//time_t timer = time(NULL);
	//tm ttt;
	//localtime_s(&ttt, &timer);
	//xml_name += "-" + to_string(ttt.tm_year + 1900)+ to_string(ttt.tm_mon + 1)+ to_string(ttt.tm_mday);
	//xml_name += to_string(ttt.tm_mon + 1);
	//xml_name += to_string(ttt.tm_mday);
	//xml_name += "-" + to_string(num);

	if (bndbox.size() > label.size()) return false;
        std::expand::create_directory(image_path);
        std::expand::create_directory(xml_path);
	std::string img_filename = filename + ".jpg";
	std::string full_path = image_path + "\\" + img_filename;
	if (!cv::imwrite(full_path, image)) return false;

	std::string xml_name = filename + ".xml";
	std::ofstream xml_file(xml_path + "\\"+xml_name);
	if (!xml_file.is_open()) return false;

	std::string folder;
	if (image_path.find_last_of("\\") != std::string::npos)
		folder = image_path.substr(image_path.find_last_of("\\") + 1);
    else if(image_path.find_last_of("/") != std::string::npos)
        folder = image_path.substr(image_path.find_last_of("/") + 1);
	
	xml_file << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
	xml_file << "<annotation>" << std::endl;
	xml_file << "    <folder>" << folder << "</folder>" << std::endl;
	xml_file << "    <filename>" << img_filename << "</filename>" << std::endl;
	xml_file << "    <path>" << full_path << "</path>" << std::endl;
	xml_file << "    <source>" << std::endl;
	xml_file << "        <database>Unknown</database>"  << std::endl;
	xml_file << "    </source>" << std::endl;
	xml_file << "    <size>" << std::endl;
	xml_file << "        <width>" << image.cols << "</width>" << std::endl;
	xml_file << "        <height>" << image.rows << "</height>" << std::endl;
	xml_file << "        <depth>" << image.channels() << "</depth>" << std::endl;
	xml_file << "    </size>" << std::endl;
	xml_file << "    <segmented>0</segmented>" << std::endl;

	for (int i = 0; i < bndbox.size(); ++i)
	{
		xml_file << "    <object>" << std::endl;
		xml_file << "        <name>" << label[i] << "</name>" << std::endl;
        xml_file << "        <pose>Unspecified</pose>" << std::endl;
		xml_file << "        <truncated>0</truncated>" << std::endl;
		xml_file << "        <difficult>0</difficult>" << std::endl;
		xml_file << "        <bndbox>" << std::endl;
		xml_file << "            <xmin>" << bndbox[i].x << "</xmin>" << std::endl;
		xml_file << "            <ymin>" << bndbox[i].y << "</ymin>" << std::endl;
		xml_file << "            <xmax>" << bndbox[i].x+bndbox[i].width << "</xmax>" << std::endl;
		xml_file << "            <ymax>" << bndbox[i].y + bndbox[i].height << "</ymax>" << std::endl;
		xml_file << "        </bndbox>" << std::endl;
		xml_file << "    </object>" << std::endl;
	}
	xml_file << "</annotation>" << std::endl;
	xml_file.close();
	xml_file.clear();

	return true;
}

//bool poisson_blender::blend_old(const cv::Mat &input_bg, const cv::Mat &input_object, const cv::Mat input_mask, cv::Mat &blended_img, cv::Rect &bounding_box)
//{
//	//bounding box and object roi (object roi > bounding box)
//	
//	cv::Mat _bin_mask;
//	binMask(input_mask, _bin_mask);
//	cv::Rect input_rect = findObjectBoundingBox(_bin_mask);
//	double rate = sqrt(input_bg.size().area() / input_rect.area() *_size_rate);
//	cv::Mat _object;
//	cv::Size _size(int(input_object.cols *rate), int(input_object.rows*rate));
//	cv::resize(input_object, _object, _size);
//	cv::Mat _mask;
//	cv::resize(_bin_mask, _mask, _size);
//	cv::Rect _bounding_box = findObjectBoundingBox(_mask);
//	int _extend_bound = 5;
//	if (_bounding_box.x - _extend_bound<0
//		|| _bounding_box.y - _extend_bound <0
//		|| _bounding_box.width + _extend_bound * 2>input_bg.cols
//		|| _bounding_box.height + _extend_bound * 2>input_bg.rows) return false;
//	cv::Rect _roi(_bounding_box.x - _extend_bound, _bounding_box.y - _extend_bound,
//		_bounding_box.width + _extend_bound * 2, _bounding_box.height + _extend_bound * 2);
//	cv::Mat object = _object(_roi).clone();
//	cv::Mat mask = _mask(_roi).clone();
//
//	cv::Rect bg_place = choicePlace(input_bg, mask);
//	bounding_box.x = bg_place.x + _extend_bound;
//	bounding_box.y = bg_place.y + _extend_bound;
//	bounding_box.width = bg_place.width- _extend_bound * 2;
//	bounding_box.height = bg_place.height - _extend_bound * 2;
//
//	//blend process
//	//blendProcess(object, input_bg, mask, blended_img);
//	cv::Mat object_grad_x, object_grad_y;
//	imgrad(object, object_grad_x, object_grad_y);
//	cv::Mat bg_grad_x, bg_grad_y;
//	imgrad(input_bg, bg_grad_x, bg_grad_y);
//
//	//copy
//	input_bg.copyTo(blended_img);
//	cv::Mat blended_img_roi = blended_img(bg_place);
//	cv::Mat blended_img_grad_x_roi = bg_grad_x(bg_place);
//	cv::Mat blended_img_grad_y_roi = bg_grad_y(bg_place);
//	object.copyTo(blended_img_roi, mask);
//	object_grad_x.copyTo(blended_img_grad_x_roi, mask);
//	object_grad_y.copyTo(blended_img_grad_y_roi, mask);
//
//	//
//	PoissonJacobiSam(blended_img_roi, blended_img_grad_x_roi, blended_img_grad_y_roi, mask, _itr, _th);
//
//	return true;
//}

//bool poisson_blender::blend_one(const cv::Mat &input_bg, const cv::Mat &input_object, const cv::Mat input_mask, 
//	cv::Mat &blended_img, cv::Rect &bounding_box, std::vector<cv::Rect> &reject_region)
//{
//	input_bg.copyTo(blended_img);
//
//    //1. cut the black edge
//    cv::Rect rect_compact = mask_maker::findMaxObjectBoundingBox(input_mask);
//    cv::Mat obj_compact = input_object(rect_compact);
//    cv::Mat mask_compact = input_mask(rect_compact);
//    cv::Size _size = get_fit_size(input_bg.size(), mask_compact.size());
//    cv::Mat _object;
//	cv::resize(obj_compact, _object, _size);
//	cv::Mat _mask;
//	cv::resize(mask_compact, _mask, _size);
//	cv::Rect _bounding_box_4_obj = mask_maker::findMaxObjectBoundingBox(_mask);
//
//    /////////////////////////////////////////////////////////
//    //no extend bdb
//    cv::Mat object = _object(_bounding_box_4_obj).clone();
//    cv::Mat mask = _mask(_bounding_box_4_obj).clone();
//    cv::Rect bg_place;
//    if (!choicePlace(blended_img.size(), mask.size(), reject_region, bg_place))
//    {
//    	return false;
//    }
//    bounding_box = bg_place;
//    /////////////////////////////////////////////////////////
//
//
//	//extend the bounding box as roi
//	//int _extend_bound = 5;
// //   if (_bounding_box.x - _extend_bound<0
// //       || _bounding_box.y - _extend_bound <0
// //       || _bounding_box.width + _extend_bound * 2>_object.cols
// //       || _bounding_box.height + _extend_bound * 2>_object.rows)
// //   {
// //       std::cout << "无法拓展区域" << std::endl;
// //       return false;
// //   }
//	//cv::Rect _roi(_bounding_box.x - _extend_bound, _bounding_box.y - _extend_bound,
//	//	_bounding_box.width + _extend_bound * 2, _bounding_box.height + _extend_bound * 2);
// //   std::cout << _object.size<<"    "<<_roi << std::endl;
// //   std::cout << _object.rows <<" x "<< _object.cols << "    "
// //       << "["<<_roi.width<<" x "<<_roi.height<<" from "
// //       <<"("<< _roi.x<<", "<<_roi.y<<")]"<< std::endl;
//	//cv::Mat object = _object(_roi).clone();
// //   std::cout << "obj clone OK" << std::endl;
//	//cv::Mat mask = _mask(_roi).clone();
// //   std::cout << "mask clone OK" << std::endl;
//	////choice place to set the object
//	//cv::Rect bg_place;
//	//if (!choicePlace(input_bg, mask, reject_region, bg_place))
//	//{
//	//	return false;
//	//}
//	//bounding_box.x = bg_place.x + _extend_bound;
//	//bounding_box.y = bg_place.y + _extend_bound;
//	//bounding_box.width = bg_place.width - _extend_bound * 2;
//	//bounding_box.height = bg_place.height - _extend_bound * 2;
//
//    //bounding_box.x = bg_place.x - _extend_bound;
//    //bounding_box.y = bg_place.y - _extend_bound;
//    //bounding_box.width = bg_place.width + _extend_bound * 2;
//    //bounding_box.height = bg_place.height + _extend_bound * 2;
//
//    ////////////////////////////////////////////////////////////
//
//	//blend process
//	cv::Mat object_grad_x, object_grad_y;
//	imgrad(object, object_grad_x, object_grad_y);
//	cv::Mat bg_grad_x, bg_grad_y;
//	imgrad(input_bg, bg_grad_x, bg_grad_y);
//	cv::Mat blended_img_roi = blended_img(bg_place);
//	cv::Mat blended_img_grad_x_roi = bg_grad_x(bg_place);
//	cv::Mat blended_img_grad_y_roi = bg_grad_y(bg_place);
//	object.copyTo(blended_img_roi, mask);
//	object_grad_x.copyTo(blended_img_grad_x_roi, mask);
//	object_grad_y.copyTo(blended_img_grad_y_roi, mask);
//
//    //////////////////////////////////////////////////
//    //融合原版
//    PoissonJacobiSam(blended_img_roi, blended_img_grad_x_roi, blended_img_grad_y_roi, mask, _itr, _th);
//    /////////////////////////////////////
//    //元龙修改
// //   cv::Mat blended_img_roi_pre_hls;
// //   cv::cvtColor(blended_img_roi, blended_img_roi_pre_hls, CV_BGR2HLS);
//	//PoissonJacobiSam(blended_img_roi, blended_img_grad_x_roi, blended_img_grad_y_roi, mask, _itr, _th);
// //   cv::Mat blended_img_roi_hls;
// //   cv::cvtColor(blended_img_roi, blended_img_roi_hls, CV_BGR2HLS);
// //   for (int i = 0; i < mask.rows; ++i)
// //   {
// //       uchar * mask_line = mask.ptr<uchar>(i);
// //       cv::Vec3b * blended_img_roi_hls_line = blended_img_roi_hls.ptr<cv::Vec3b>(i);
// //       cv::Vec3b * blended_img_roi_pre_hls_line = blended_img_roi_pre_hls.ptr<cv::Vec3b>(i);
// //       for (int j = 0; j < mask.cols; ++j)
// //       {
// //           //blended_img_roi_hls.at<cv::Vec3b>(i, j)[0] = (int(blended_img_roi_hls.at<cv::Vec3b>(i, j)[0]) + 
// //           //    blended_img_roi_pre_hls.at<cv::Vec3b>(i, j)[0]) / 2;
// //           //if(mask.at<uchar>(i, j)!=0)
// //           //    blended_img_roi_hls.at<cv::Vec3b>(i, j)[0] = blended_img_roi_pre_hls.at<cv::Vec3b>(i, j)[0];
// //           if (mask_line[j] != 0)
// //           {
// //               blended_img_roi_hls_line[j][0] = blended_img_roi_pre_hls_line[j][0];
// //           }
// //       }
// //   }
// //   cv::cvtColor(blended_img_roi_hls, blended_img_roi_hls, CV_HLS2BGR);
// //   blended_img_roi_hls.copyTo(blended_img_roi, mask);
//    /////////////////////////////////////
//	return true;
//}

bool poisson_blender::blend_one(const cv::Mat &input_bg, const cv::Mat &input_object,
    cv::Mat &blended_img, cv::Rect &bounding_box, std::vector<cv::Rect> &reject_region)
{        

    //clock_t begin, end;
    //std::cout << "pre = ";
    //begin = clock();

    input_bg.copyTo(blended_img);
    
    cv::Mat obj_bgr, obj_a;


    split_png(input_object, obj_bgr, obj_a);

    cv::Mat input_mask;
    cv::threshold(obj_a, input_mask, 0, 255, CV_THRESH_BINARY);

    //1. cut the black edge
    cv::Rect rect_compact = findMaskBoundingBox(input_mask);
    cv::Mat obj_bgr_compact = obj_bgr(rect_compact);
    cv::Mat obj_a_compact = obj_a(rect_compact);
    cv::Mat mask_compact = input_mask(rect_compact);
    cv::Size _size = get_fit_size(blended_img.size(), rect_compact.size());

    cv::Mat object_bgr, object_a, mask;
    cv::resize(obj_bgr_compact, object_bgr, _size);
    cv::resize(obj_a_compact, object_a, _size);
    cv::resize(mask_compact, mask, _size);

    /////////////////////////////////////////////////////////
    //no extend bdb
    cv::Rect bg_place;
    //TODO
    //if (!choicePlace(blended_img.size(), _size, reject_region, bg_place))
    //{
    //    return false;
    //}
    if (!choicePlace2Cover(blended_img.size(), _size, reject_region, bg_place,0.1,0.2))
    {
        return false;
    }

    bounding_box = bg_place;

    //blend process
    cv::Mat object_grad_x, object_grad_y;
    imgrad(object_bgr, object_grad_x, object_grad_y);

    //expand and check
    cv::Rect blended_img_imgrad_roi_extend(bg_place.x - 10,bg_place.y - 10, bg_place.width + 20, bg_place.height + 20);
    if (blended_img_imgrad_roi_extend.x < 0) { blended_img_imgrad_roi_extend.x = 0; }
    if (blended_img_imgrad_roi_extend.y < 0) { blended_img_imgrad_roi_extend.y = 0; }
    if (blended_img_imgrad_roi_extend.width > blended_img.cols - blended_img_imgrad_roi_extend.x)
    {
        blended_img_imgrad_roi_extend.width = blended_img.cols - blended_img_imgrad_roi_extend.x;
    }
    if (blended_img_imgrad_roi_extend.height > blended_img.rows - blended_img_imgrad_roi_extend.y)
    {
        blended_img_imgrad_roi_extend.height = blended_img.rows - blended_img_imgrad_roi_extend.y;
    }

    //roi in roi 
    cv::Rect blended_img_imgrad_roi_extend_roi = bg_place - blended_img_imgrad_roi_extend.tl();
    cv::Mat bg_grad_x_extend, bg_grad_y_extend;
    imgrad(blended_img(blended_img_imgrad_roi_extend), bg_grad_x_extend, bg_grad_y_extend);
    cv::Mat blended_img_roi = blended_img(bg_place);
    cv::Mat blended_img_grad_x_roi = bg_grad_x_extend(blended_img_imgrad_roi_extend_roi);
    cv::Mat blended_img_grad_y_roi = bg_grad_y_extend(blended_img_imgrad_roi_extend_roi);

    for (int i = 0; i < blended_img_roi.rows; ++i)
    {
        cv::Vec3b * p_row_object_bgr = object_bgr.ptr<cv::Vec3b>(i);
        cv::Vec3b * p_row_blended_img_roi = blended_img_roi.ptr<cv::Vec3b>(i);
        cv::Vec3f * p_row_object_grad_x = object_grad_x.ptr<cv::Vec3f>(i);
        cv::Vec3f * p_row_blended_img_grad_x_roi = blended_img_grad_x_roi.ptr<cv::Vec3f>(i);
        cv::Vec3f * p_row_object_grad_y = object_grad_y.ptr<cv::Vec3f>(i);
        cv::Vec3f * p_row_blended_img_grad_y_roi = blended_img_grad_y_roi.ptr<cv::Vec3f>(i);
        uchar * p_row_object_a = object_a.ptr<uchar>(i);
        for (int j = 0; j < blended_img_roi.cols; ++j)
        {
            if (p_row_object_a[j] == 255)
            {
                p_row_blended_img_roi[j] = p_row_object_bgr[j];
                p_row_blended_img_grad_x_roi[j] = p_row_object_grad_x[j];
                p_row_blended_img_grad_y_roi[j] = p_row_object_grad_y[j];
            }
            else if(p_row_object_a[j] == 0)
            {

            }
            else
            {
                double weight = p_row_object_a[j] / 255.0;
                p_row_blended_img_roi[j] = p_row_object_bgr[j] * weight + p_row_blended_img_roi[j] * (1 - weight);
                p_row_blended_img_grad_x_roi[j] = p_row_object_grad_x[j] * weight + p_row_blended_img_grad_x_roi[j] * (1 - weight);
                p_row_blended_img_grad_y_roi[j] = p_row_object_grad_y[j] * weight + p_row_blended_img_grad_y_roi[j] * (1 - weight);
            }
        }
    }

    //end = clock();
    //std::cout << double(end - begin) / CLOCKS_PER_SEC * 1000 << "    ";

    //////////////////////////////////////////////////
    //融合原版

    //begin = clock();
    
    //cv::Mat blended_mask(blended_img.size(), CV_8UC1);
    //blended_mask.setTo(0);
    //mask.copyTo(blended_mask(bounding_box));
    //cv::illuminationChange(blended_img, blended_mask, blended_img);
    //cv::waitKey(0);


    //PoissonJacobiSam(blended_img_roi, blended_img_grad_x_roi, blended_img_grad_y_roi, mask, _itr, _th);
    PoissonJacobiSam(blended_img_roi, blended_img_grad_x_roi, blended_img_grad_y_roi, mask, 10, _th);

    //cv::seamlessClone(object_bgr.clone(), blended_img_roi.clone(), mask.clone(), cv::Point(mask.cols / 2, mask.rows / 2), blended_img_roi, cv::NORMAL_CLONE);

    //cv::Mat temp_obj = object_bgr.clone();
    //blended_img_roi.copyTo(temp_obj, mask);

    ////cv::imshow("blended_img_roi", blended_img_roi);

    ////cv::Mat  t_blended_img_roi = input_bg(bounding_box).clone();
    ////cv::Mat t_blended_img_roi = blended_img_roi.clone();

    //cv::imshow("temp_obj", temp_obj);
    //cv::imshow("blended_img_roi", blended_img_roi);
    //cv::imshow("mask", mask);

    //cv::seamlessClone(temp_obj, input_bg(bounding_box).clone(), mask.clone(), cv::Point(blended_img_roi.cols / 2, blended_img_roi.rows / 2), blended_img_roi,cv::NORMAL_CLONE);

    //cv::imshow("out_blended_img_roi", blended_img_roi);
    //cv::waitKey(0);

    //cv::Mat t_bg = input_bg.clone();
    //t_blended_img_roi.copyTo(t_bg(bounding_box), mask);
    //t_bg(bounding_box).copyTo(blended_img_roi);


    //for (int i = 0; i < blended_img_roi.rows; ++i)
    //{
    //    cv::Vec3b * p_t_blended_img_roi = t_blended_img_roi.ptr<cv::Vec3b>(i);
    //    cv::Vec3b * p_input_bg = blended_img_roi.ptr<cv::Vec3b>(i);
    //    cv::Vec3b * p_blended_img_roi = blended_img_roi.ptr<cv::Vec3b>(i);
    //    uchar * p_row_object_a = object_a.ptr<uchar>(i);
    //    for (int j = 0; j < blended_img_roi.cols; ++j)
    //    {
    //        if (p_row_object_a[j] == 255)
    //        {
    //            p_blended_img_roi[j] = p_t_blended_img_roi[j];
    //        }
    //        else if (p_row_object_a[j] == 0)
    //        {
    //            p_blended_img_roi[j] = p_input_bg[j];
    //        }
    //        else
    //        {
    //            double weight = p_row_object_a[j] / 255.0;
    //            p_blended_img_roi[j] = p_t_blended_img_roi[j] * weight + p_input_bg[j] * (1 - weight);

    //        }
    //    }
    //}







    //cv::imshow("t_blended_img_roi", t_bg(bounding_box));
    //t_blended_img_roi.copyTo(blended_img_roi, mask);
    //cv::imshow("t_blended_img_roi", t_blended_img_roi);

    //cv::waitKey(0);


    //end = clock();
    //std::cout <<"mix = "<< double(end - begin) / CLOCKS_PER_SEC * 1000 << "    " << std::endl;

    /////////////////////////////////////
    //元龙修改
    //cv::Mat blended_img_roi_pre_hls;
    //cv::cvtColor(blended_img_roi, blended_img_roi_pre_hls, CV_BGR2HLS);
    //PoissonJacobiSam(blended_img_roi, blended_img_grad_x_roi, blended_img_grad_y_roi, mask, _itr, _th);
    //cv::Mat blended_img_roi_hls;
    //cv::cvtColor(blended_img_roi, blended_img_roi_hls, CV_BGR2HLS);
    //for (int i = 0; i < mask.rows; ++i)
    //{
    //    uchar * p_row_input_mask = mask.ptr<uchar>(i);
    //    cv::Vec3b * p_row_blended_img_roi_hls = blended_img_roi_hls.ptr<cv::Vec3b>(i);
    //    cv::Vec3b * p_row_blended_img_roi_pre_hls = blended_img_roi_pre_hls.ptr<cv::Vec3b>(i);
    //    for (int j = 0; j < mask.cols; ++j)
    //    {
    //        //blended_img_roi_hls.at<cv::Vec3b>(i, j)[0] = (int(blended_img_roi_hls.at<cv::Vec3b>(i, j)[0]) + 
    //        //    blended_img_roi_pre_hls.at<cv::Vec3b>(i, j)[0]) / 2;
    //        if (p_row_input_mask[j] != 0)
    //            p_row_blended_img_roi_hls[j][0] = p_row_blended_img_roi_pre_hls[j][0];
    //    }
    //}
    //cv::cvtColor(blended_img_roi_hls, blended_img_roi_hls, CV_HLS2BGR);
    //blended_img_roi_hls.copyTo(blended_img_roi, mask);


    /////////////////////////////////////
    return true;
}

//bool poisson_blender::blend(const cv::Mat &input_bg,
//	const std::vector<cv::Mat> &input_object,
//	const std::vector<cv::Mat> &input_mask,
//	cv::Mat &blended_img,
//	std::vector<cv::Rect> &bounding_box,
//	std::vector<bool> &is_blended)
//{
//    bool flag = true;
//
//	is_blended.resize(input_object.size(),false);
//	int n = input_object.size() < input_mask.size() ? input_object.size() : input_mask.size();
//	
//	std::vector<cv::Rect> reject_region;
//	cv::Mat temp_bg;
//	input_bg.copyTo(temp_bg);
//	cv::Rect one_bounding_box;
//	bounding_box.resize(0);
//	for (int i = 0; i < n; ++i)
//	{
//		is_blended[i] = blend_one(temp_bg, input_object[i], input_mask[i], blended_img, one_bounding_box, reject_region);
//        if (!is_blended[i])
//        {
//            //std::cout << "blend_one 出错啦" << std::endl;
//            flag = false;
//            break;
//        }
//		bounding_box.push_back(one_bounding_box);
//		blended_img.copyTo(temp_bg);
//	}
//    if (!flag)
//    {
//        //std::cout << "返回了 false" << std::endl;
//        return false;
//    }
//	else return true;
//}


//bool poisson_blender::blend(const std::string &input_bg, const std::vector<std::string> &input_object,
//	const std::vector<std::string> &input_mask, cv::Mat &blended_img, 
//	std::vector<cv::Rect> &bounding_box, std::vector<bool> &is_blended)
//{
//	cv::Mat mat_input_bg = cv::imread(input_bg);
//	if (mat_input_bg.empty()) return false;
//	
//	cv::Mat t;
//
//	std::vector<cv::Mat> v_mat_input_object;
//	for (int i = 0; i < input_object.size(); ++i)
//	{
//		t = cv::imread(input_object[i]);
//		if (t.empty()) return false;
//		v_mat_input_object.push_back(t);
//	}
//
//	std::vector<cv::Mat> v_mat_input_mask;
//	for (int i = 0; i < input_mask.size(); ++i)
//	{
//		t = cv::imread(input_mask[i],CV_LOAD_IMAGE_GRAYSCALE);
//        //t = mask_maker::readMask(input_mask[i]);
//		if (t.empty()) return false;
//		v_mat_input_mask.push_back(t);
//	}
//
//	return blend(mat_input_bg, v_mat_input_object, v_mat_input_mask, blended_img, bounding_box, is_blended);
//}


bool poisson_blender::blend(const cv::Mat &input_bg, const std::vector<cv::Mat> &input_object, cv::Mat &blended_img,
    std::vector<cv::Rect> &bounding_box, std::vector<bool> &is_blended)
{
    bool flag = true;

    int n = input_object.size();
    is_blended.resize(n, false);

    std::vector<cv::Rect> reject_region;
    cv::Mat temp_bg;
    input_bg.copyTo(temp_bg);
    cv::Rect one_bounding_box;
    bounding_box.resize(0);

    //clock_t begin, end;
    //std::cout << "Aug = ";
    for (int i = 0; i < n; ++i)
    {

        //begin = clock();
        is_blended[i] = blend_one(temp_bg, input_object[i], blended_img, one_bounding_box, reject_region);

        if (!is_blended[i])
        {
            //std::cout << "blend_one 出错啦" << std::endl;
            flag = false;
            break;
        }
        bounding_box.push_back(one_bounding_box);
        blended_img.copyTo(temp_bg);
        //end = clock();
        //std::cout << double(end - begin) / CLOCKS_PER_SEC * 1000 << "  ";
    }

    //std::cout << std::endl;

    if (!flag)
    {
        //std::cout << "返回了 false" << std::endl;
        return false;
    }
    else return true;
}


bool poisson_blender::blend(const std::string &input_bg, const std::vector<std::string> &input_object, cv::Mat &blended_img,
    std::vector<cv::Rect> &bounding_box, std::vector<bool> &is_blended)
{
    //clock_t begin, end;
    //begin = clock();

    cv::Mat mat_input_bg = cv::imread(input_bg);
    if (mat_input_bg.empty()) return false;

    cv::Mat t;

    std::vector<cv::Mat> v_mat_input_object;
    for (int i = 0; i < input_object.size(); ++i)
    {
        t = cv::imread(input_object[i], CV_LOAD_IMAGE_UNCHANGED);
        if (t.empty() || t.channels() != 4) return false;
        v_mat_input_object.push_back(t);
    }

    //end = clock();
    //std::cout << "read = "<<double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms    ";

    //begin = clock();
    bool flag = blend(mat_input_bg, v_mat_input_object, blended_img, bounding_box, is_blended);
    //end = clock();
    //std::cout << "aug = "<<double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms    ";
    //std::cout << std::endl;
    return flag;
}

void poisson_blender::blend_thread(const cv::Mat *input_bg, const std::vector<cv::Mat> *input_object, cv::Mat *blended_img,
    std::vector<cv::Rect> *bounding_box, std::vector<bool> *is_blended, poisson_blender * p_this, int * p_flag)
{
    bool flag = true;

    int n = (*input_object).size();
    (*is_blended).resize(n, false);

    std::vector<cv::Rect> reject_region;
    cv::Mat temp_bg;

    //cv::imshow("input_bg", *input_bg);
    //cv::waitKey(0);

    (*input_bg).copyTo(temp_bg);
    cv::Rect one_bounding_box;
    (*bounding_box).resize(0);

    //clock_t begin, end;
    //std::cout << "Aug = ";
    for (int i = 0; i < n; ++i)
    {

        //begin = clock();
        (*is_blended)[i] = p_this->blend_one(temp_bg, (*input_object)[i], (*blended_img), one_bounding_box, reject_region);

        
        if (!(*is_blended)[i])
        {
            //std::cout << "blend_one 出错啦" << std::endl;
            flag = false;
            break;
        }
        (*bounding_box).push_back(one_bounding_box);
        (*blended_img).copyTo(temp_bg);
        //end = clock();
        //std::cout << double(end - begin) / CLOCKS_PER_SEC * 1000 << "  ";
        //std::cout << i << std::endl;
    }

    //std::cout << *p_flag << std::endl;

    if (!flag)
    {
        //std::cout << "返回了 false" << std::endl;
        *p_flag = 0;
        //std::cout << *p_flag << std::endl;
        return ;
    }
    else
    {
        *p_flag = 1;
        //std::cout << *p_flag << std::endl;
        return;
    }
}



//void poisson_blender::blend_thread(const cv::Mat &mat_input_bg, const std::vector<cv::Mat> &v_mat_input_object, cv::Mat &blended_img,
//    std::vector<cv::Rect> &bounding_box, std::vector<bool> &is_blended,int &flag_return)
//{
//    //begin = clock();
//    //cv::Mat  t_mat_input_bg;
//    //cv::resize(mat_input_bg, t_mat_input_bg, cv::Size(640, 480));
//    //cv::imshow("mat_input_bg", t_mat_input_bg);
//    //cv::waitKey(0);
//
//    //blend_thread_cal(&mat_input_bg, &v_mat_input_object, &blended_img, &bounding_box, &is_blended, p_pack);
//
//    //end = clock();
//    //std::cout << "aug = "<<double(end - begin) / CLOCKS_PER_SEC * 1000 << "ms    ";
//    //std::cout << std::endl;
//     return;
//}





//bool poisson_blender::blend(const cv::Mat &input_object, const cv::Mat &input_bg, cv::Mat &blended_img, cv::Rect &bounding_box)
//{
//	//mask
//	cv::Mat input_mask;
//	//checkoutMask(input_object, input_mask);
//
//	return blend(input_object, input_bg, input_mask, blended_img, bounding_box);
//}

