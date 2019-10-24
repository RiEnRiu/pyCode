#ifndef MASK_MAKER_GUI_H
#define MASK_MAKER_GUI_H


#include <opencv2/core/core.hpp>

#include<vector>
#include<string>
#include<unordered_map>


enum CONFIG_KEY
{
    CONFIG_KEY_bg_threshold = 0,
    CONFIG_KEY_fg_threshold = 1,
    CONFIG_KEY_edge_smooth_kernal_size = 2,
    CONFIG_KEY_mask_threshold = 3
};

struct mask_config
{
    std::string obj_path;
    std::unordered_map<int, int> param;

    std::vector<std::vector<cv::Point>> ___transparent;
    std::vector<std::vector<cv::Point>> ___invariant;
    std::vector<std::vector<cv::Point>> ___erase_area;

    //std::vector<cv::Rect> transparent;
    //std::vector<cv::Rect> invariant;
    //std::vector<cv::Rect> erase_area;
};

class mask_maker
{
private:
    //static std::vector<cv::Mat> temp_mat;
    std::string ___config_save_path;
    static double rate_param;
    static mask_config config_default;

    //private
    //Matte
    //degree

    bool isAngleInRange(double angle, double minRange, double maxRange);
    double angleABS(double angle1, double angle2);
    cv::Mat getTripMap(const cv::Mat &fit_obj_rgb_8uc3, int bg_threshold, int fg_threshold);
    cv::Mat outlineMatting(const cv::Mat &fit_obj_rgb_8uc3, int bg_threshold, int fg_threshold, int edge_smooth_kernal_size, int mask_threshold);
    void cleanBgHue(const cv::Mat &fit_obj_rgb_8uc3, const std::vector<std::vector<cv::Point>> &fit_invariant, const cv::Mat &fit_outline, std::vector<cv::Mat> &v_bgr_output);
    cv::Mat alphaMatting(const cv::Mat &fit_obj_rgb_8uc3, const std::vector<std::vector<cv::Point>> &fit_transparent, const cv::Mat &fit_outline);
    cv::Mat Matting(const std::string &obj_path, const mask_config &config);
    cv::Mat Matting(const mask_config &config);
    static cv::Mat getInvariantPattern(const cv::Mat &input);
    bool isIntersect(const cv::Rect &region1, const cv::Rect &region2);

    //void_ptr_pack[0] = &mask_make;
    //void_ptr_pack[1] = &mask_config;
    //void_ptr_pack[2] = &bg_brightness;
    //void_ptr_pack[3] = &effect_mat;
    static void on_Trackbar_config_value_0(int bar_value, void *void_ptr_pack);
    static void on_Trackbar_config_value_1(int bar_value, void *void_ptr_pack);
    static void on_Trackbar_config_value_2(int bar_value, void *void_ptr_pack);
    static void on_Trackbar_config_value_3(int bar_value, void *void_ptr_pack);
    static void on_Trackbar_config_value_4(int bar_value, void *void_ptr_pack);
    //void_ptr_pack[0] = &mask_make;
    //void_ptr_pack[1] = &mask_config;
    //void_ptr_pack[2] = &add_or_delete_flag;int = 0, add; int = 1, delete;
    //void_ptr_pack[3] = &bg_brightness;
    //void_ptr_pack[4] = &effect_mat;
    //void_ptr_pack[5] = &is_shift_obj
    //static void on_Mouse_left_btn(int event, int x, int y, int flags, void *void_ptr_pack);
    static void on_Mouse(int event, int x, int y, int flags, void *void_ptr_pack);
    //static void on_Trackbar_config_value_4(int bar_value, void *v_p_config);

    int makeConfigGUI(mask_config &inputOutput_config);

    bool str2point8(const std::string &config_str, std::vector<cv::Point> &P8);
    bool point82str(const std::vector<cv::Point> &P8, std::string &config_str);

    cv::Mat p8AreaMask(const cv::Size &src_size, const std::vector<std::vector<cv::Point>> &areas);
    cv::Mat p8AreaInvMask(const cv::Size &src_size, const std::vector<std::vector<cv::Point>> &areas);
    void p8Resize(const std::vector<cv::Point> &src, std::vector<cv::Point> &dst, double x_rate, double y_rate);

    //bool str2rect(const std::string &config_str, cv::Rect &rect);
    //bool rect2str(const cv::Rect &rect, std::string &config_str);

public:

    //useful method
    static cv::Rect findMaxObjectBoundingBox(const cv::Mat &bin_mask_8UC1);
    static cv::Rect findMaskBoundingBox(const cv::Mat &bin_mask_8UC1);
    static void split_8UC4(const cv::Mat &bgra, cv::Mat &bgr, cv::Mat &a);
    static void merge(const cv::Mat &bgr, const cv::Mat &a, cv::Mat &bgra);
    static cv::Mat png2binMask(const cv::Mat &bgra);
    static cv::Mat compactPNG(const cv::Mat &bgra);
    static void imshow_png(const std::string &windowsName, const cv::Mat &mat_8uc4, double bg_brightness = 1.0);

    //param
    mask_maker(const std::string &_config_save_path = "./aug/config/MASK_CONFIG");

    ~mask_maker();

    void set_config_save_path(const std::string &_config_save_path);

    //run gui
    bool make_config(const std::vector<std::vector<std::string>> &obj_paths, std::vector<std::vector<mask_config>> &config, const std::vector<mask_config> &loaded_config =std::vector<mask_config>());
    //void make_config(std::vector<std::string> obj_paths, std::vector<mask_config> &config);

    //analysis config with boost
    bool load_config(const std::string &config_path, std::vector<mask_config> &config);

    //write in "___config_save_path"
    bool save_config(const std::vector<std::vector<mask_config>> &config);
    bool save_config(const std::vector<mask_config> &config);

    //create one by one
    //void make_mask(const mask_config &config, cv::Mat &obj, cv::Mat &mask);
    cv::Mat make_mask(const mask_config &config);

    //gui show
    bool preview_mask(mask_config &inputOutput_config, double bg_brightness, cv::Mat &view_mat);
    //bool preview_mask(mask_config &inputOutput_config, double bg_brightness);

};



#endif
