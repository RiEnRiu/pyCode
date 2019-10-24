// view_mask.cpp : 定义控制台应用程序的入口点。
//


#include"aug_solver/mask_maker/mask_maker_gui.h"
#include"stl_expand/stl_expand.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include<fstream>
#include<string>
#include<vector>

#include<boost/program_options.hpp>  
//#include<boost/filesystem.hpp>
//#include<boost/algorithm/string.hpp>
//namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace std;
using namespace cv;


vector<string> obj_list;
vector<string> mask_list;
vector<string> label_list;
int global_index = 0;
double mask_brightness = 1.0;
int mask_rate_bar = 100;

//bool get_image_names(const std::string& dir, std::vector<std::string>& filenames)
//{
//    fs::path path(dir);
//    if (!fs::exists(path))
//    {
//        return false;
//    }
//
//    fs::directory_iterator end_iter;
//    //filenames.resize(0);
//    for (fs::directory_iterator iter(path); iter != end_iter; ++iter)
//    {
//        if (fs::is_regular_file(iter->status()))
//        {
//            std::string ext = iter->path().extension().string();
//            boost::algorithm::to_lower(ext);
//            if (ext == ".png")
//                filenames.push_back(iter->path().string());
//        }
//        if (fs::is_directory(iter->status()))
//        {
//            get_image_names(iter->path().string(), filenames);
//        }
//    }
//    return true;
//}

void on_trackbar_share()
{
    Mat obj;
    obj = imread(obj_list[global_index], CV_LOAD_IMAGE_UNCHANGED);

    if (obj.empty()||obj.channels()!=4)
    {
        cout << obj_list[global_index] <<": is not a (png) image ." << endl;
        exit(-1);
    }

    double rate = 1;
    rate = sqrt(750.0 * 500.0 / obj.cols / obj.rows);
    cv::Size fit_size(obj.cols*rate, obj.rows*rate);
    cv::resize(obj, obj, fit_size);

    mask_maker::imshow_png("obj", obj, mask_brightness);
    return;
}

void on_trackbar_mask_rate(int _mask_rate, void *p)
{
    mask_rate_bar = _mask_rate;
    on_trackbar_share();
}

void on_trackbar_index_rate(int _index_rate, void *p)
{
    global_index = int(_index_rate / 100.0 * (obj_list.size() - 1));
    on_trackbar_share();
}

void on_trackbar_mask_brightness(int _mask_brightness, void *p)
{
    mask_brightness = double(_mask_brightness) / 100.0;
    on_trackbar_share();
}



int main(int argc, char **argv)
{
    


    //cv::Mat img = cv::imread("F:\\ver3.0.1\\obj\\bs-bskl-gz-yw-330ml\\0.jpg",0);
    //cv::resize(img, img, Size(750, 500));



    //cv::Mat cornerStrength;
    //cornerHarris(img, cornerStrength, 2, 3, 0.01);

    //Mat harrisConrner;
    //threshold(cornerStrength, harrisConrner, 0.00001, 255,THRESH_BINARY);


    //imshow("cornerStrength", harrisConrner);




    //cv::Mat red_mat(100, 100, CV_8UC3);
    //red_mat.setTo(cv::Scalar(0, 100,0));
    //cv::imshow("red_mat",red_mat);

    //cv::Mat red_hls;
    //cv::cvtColor(red_mat, red_hls, CV_BGR2HLS);
    //for (int i = 0; i < red_hls.rows; ++i)
    //{
    //    for (int j = 0; j < red_hls.cols; ++j)
    //    {
    //        //red_hls.at<Vec3b>(i, j)[1] = 255;
    //        std::cout << (int)red_hls.at<Vec3b>(i, j)[2] << std::endl;
    //        //red_hls.at<Vec3b>(i, j)[2] = 255;
    //    }
    //}
    //cv::Mat changed_mat;
    //cv::cvtColor(red_hls, changed_mat, CV_HLS2BGR);
    //cv::imshow("changed_mat", changed_mat);

    //cv::waitKey(0);


    po::options_description desc("view png");
    string list_str;
    string dir_str;
    desc.add_options()
        ("list", po::value<string>(&list_str), "[obj path] [label] list file (saved from Augmentor.exe).")
        ("dir", po::value<string>(&dir_str), "Scan all images (png) to show.");

    //fs::path p("G:/image/aug/aug");
    //cout << p.filename().string() << endl;
    //cout << fs::is_directory(p) << endl;
    //cout << fs::is_regular_file(p) << endl;
    //cout << fs::is_regular(p) << endl;
    //getchar();

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
        return -1;
    }

    //dir_str = "G:\\ver3.0.2\\aug_obj1_2018_06_05";

    if (list_str.size() != 0)
    {
        vector<vector<string>> part;
        std::expand::check_out_list(list_str, " ", part);
        if (part.size() != 2) { cout << desc << endl; return -1; }
        obj_list = part[0];
        label_list = part[1];
    }
    else if (dir_str.size() != 0)
    {
        std::vector<std::vector<std::string>> _temp_scan_file_list_output;
        std::expand::scan_file_list(dir_str, ".png", obj_list, std::expand::SCAN_ROOT_PATH_FILE_FULL_NAME,_temp_scan_file_list_output);
        label_list.resize(obj_list.size(), "");
    }
    if (obj_list.size() == 0)
    {
        cout << desc << endl;
        return -1;
    }

    if (obj_list.size() == 0)
    {
        cout << "No image can be read." << endl;
        return -1;
    }

    int t_index_rate = 0;

    int t_mask_brightness = 100;

    int key_value = 0;

    while (1)
    {
        namedWindow("obj", WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
        createTrackbar("预览位置", "obj", &t_index_rate, 100, on_trackbar_index_rate);
        createTrackbar("预览调整", "obj", &t_mask_brightness, 100, on_trackbar_mask_brightness);

        on_trackbar_share();
        key_value = waitKey(0);
        if (key_value == 'd' || key_value == 'D')
        {
            global_index = (global_index + 1) % obj_list.size();
            if (obj_list.size() == 1) { t_index_rate = 0; }
            else { t_index_rate = global_index * 100 / (obj_list.size() - 1); }
        }
        if (key_value == 'a' || key_value == 'A')
        {
            global_index = (global_index - 1 + obj_list.size()) % obj_list.size();
            if (obj_list.size() == 1) { t_index_rate = 0; }
            else { t_index_rate = global_index * 100 / (obj_list.size() - 1); }
        }
        if (key_value == 'W'|| key_value == 'w')
        {
            t_index_rate = (t_index_rate + 1);
            if (t_index_rate == 101) t_index_rate = 100;
            else if (t_index_rate == -1)t_index_rate = 0;
            global_index = int(t_index_rate / 100.0 * (obj_list.size() - 1));
        }
        if (key_value == 's' || key_value == 'S')
        {
            t_index_rate = (t_index_rate - 1);
            if (t_index_rate == 101) t_index_rate = 100;
            else if (t_index_rate == -1)t_index_rate = 0;
            global_index = int(t_index_rate / 100.0 * (obj_list.size() - 1));
        }
        if (key_value == 27) break;
    }
    destroyAllWindows();
    return 0;
}

