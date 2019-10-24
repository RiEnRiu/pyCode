// Augmentor.cpp : 定义控制台应用程序的入口点。
//
#include"aug_solver.h"
#include"aug_function.h"

#include"mask_maker/mask_maker_gui.h"
#include"stl_expand/stl_expand.hpp"

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<ctime>
#include<sstream>
#include<fstream>

#include<iostream>
//#include<io.h>
//#include<direct.h>
#include<thread>


//#include<boost\filesystem.hpp>
//namespace fs = boost::filesystem;

#include <boost/program_options.hpp> 
namespace po = boost::program_options;

bool aug_png_slover::augment_one_mat_obj(const std::string &obj, std::vector<cv::Mat> &aug_obj, const std::vector<___aug_operation> &step_opt)
{
    bool have_NotBackup = false;
    bool have_Pyramid = false;
    cv::Mat obj_mat = cv::imread(obj,CV_LOAD_IMAGE_UNCHANGED);
    aug_obj.resize(0);
    for (int i = 0; i < step_opt.size(); ++i)
    {
        std::vector<cv::Mat> obj_aug_one_opt;
        std::vector<double> rand_param;
        const ___aug_operation &opt = step_opt[i];
        if (opt.opt_name == "Rotate")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imRotate(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imRotate(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "Flip")
        {
            augmentor_png::imFlip(obj_mat, obj_aug_one_opt);
        }
        else if (opt.opt_name == "FlipUD")
        {
            augmentor_png::imFlipUD(obj_mat, obj_aug_one_opt);
        }
        else if (opt.opt_name == "FlipLR")
        {
            augmentor_png::imFlipLR(obj_mat, obj_aug_one_opt);
        }
        else if (opt.opt_name == "Crop")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imCrop(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imCrop(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "Affine")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imAffine(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imAffine(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "AffineX")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imAffineX(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imAffineX(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "AffineY")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imAffineY(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imAffineY(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "Noise")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imAddNoise_gauss(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imAddNoise_gauss(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "Hue")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imHue(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imHue(obj_mat, obj_aug_one_opt, opt.param);
            }
        }

        else if (opt.opt_name == "Saturation")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imSaturation(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imSaturation(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "Lightness")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imLightness(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imLightness(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "PerspectiveUL")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspectiveUL(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspectiveUL(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "PerspectiveU")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspectiveU(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspectiveU(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "PerspectiveUR")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspectiveUR(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspectiveUR(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "PerspectiveL")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspectiveL(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspectiveL(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "Perspective")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspective(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspective(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "PerspectiveR")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspectiveR(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspectiveR(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "PerspectiveDL")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspectiveDL(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspectiveDL(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "PerspectiveD")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspectiveD(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspectiveD(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "PerspectiveDR")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imPerspectiveDR(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imPerspectiveDR(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "Distort")
        {
            if (opt.is_random_param)
            {
                get_rand(opt.param, rand_param);
                augmentor_png::imDistort(obj_mat, obj_aug_one_opt, rand_param);
            }
            else
            {
                augmentor_png::imDistort(obj_mat, obj_aug_one_opt, opt.param);
            }
        }
        else if (opt.opt_name == "Pyramid")
        {
            augmentor_png::imPyramid(obj_mat, obj_aug_one_opt, opt.param[0], opt.param[1]);
            have_Pyramid = true;
        }
        else if (opt.opt_name == "NotBackup")
        {
            have_NotBackup = true;
        }
        else
        {
            return false;
        }
        aug_obj.insert(aug_obj.end(), obj_aug_one_opt.begin(), obj_aug_one_opt.end());
    }

    if ((!have_NotBackup)&&(!have_Pyramid))
    {
        aug_obj.push_back(obj_mat);
    }

    return true;
}

aug_png_slover::aug_png_slover(const std::string &_aug_config_path)
{
    set_aug_config(_aug_config_path);
}

aug_png_slover::~aug_png_slover() {}

bool aug_png_slover::set_aug_config(const std::string &_aug_config_path)
{
    aug_config_file_path = _aug_config_path;
    is_load_opt = checkOutOperation();
    return is_load_opt;
}

bool aug_png_slover::set_aug_config(const std::vector<std::vector<___aug_operation>>  &_aug_config)
{
    is_load_opt = checkOperation(_aug_config);
    return is_load_opt;
}

bool aug_png_slover::checkOutOperation()
{
    std::ifstream file(aug_config_file_path);
    if (!file.is_open())
    {
        std::cout << "check out operation error, can't find file" << std::endl;
        return false;
    }
    std::unordered_map<std::string, std::string> all_step;
    while(!file.eof())
    {
        std::string line;
        std::getline(file, line);
        line = line.substr(0, line.find_first_of('#'));
        if (line.size() == 0) continue;
        std::string step,step_command;
        std::expand::split2(line, "=", step, step_command);
        if (step.size() != 0 && step_command.size() != 0)
        {
            all_step[step] = step_command;
        }
    }

    operations.resize(0);

    for (int i = 0; i < all_step.size(); ++i)
    {
        std::string step_key = "step";
        step_key += std::to_string(i + 1);
        auto itr = all_step.find(step_key);
        if (itr == all_step.end()) break;
        std::vector<___aug_operation> one_aug_opt;
        if(!checkOutOperation_line(itr->second, one_aug_opt)) return false;
        operations.push_back(one_aug_opt);
    }
   
    if (!checkOperation(operations)) return false;
    printOperation(operations);
    return true;
}

bool aug_png_slover::checkOutOperation_one(const std::string &opt_str, ___aug_operation &opt)
{
    std::string fun_name;
    std::string param_in_fun;
    std::expand::split2(opt_str, "[(", fun_name, param_in_fun);
    std::expand::eliminate(fun_name, " ", opt.opt_name);
    if (param_in_fun.size() == 0)
    {
        opt.is_random_param = false;
        return true;
    }
    param_in_fun = param_in_fun.substr(0, param_in_fun.find_last_of(")]"));


    size_t rand_index = param_in_fun.find("rand");
    if (rand_index == std::string::npos)
    {
        std::vector<std::string> number_param;
        std::expand::split(param_in_fun, ",; ", number_param);
        opt.is_random_param = false;
        opt.param.resize(number_param.size());
        for (int i = 0; i < number_param.size(); ++i)
        {
            opt.param[i] = std::stod(number_param[i]);
        }
    }
    else
    {
        opt.is_random_param = true;
        std::string rand_name;
        std::string rand_param_str;
        //rand param;
        std::expand::split2(opt_str, "[(", rand_name, rand_param_str);
        if (rand_param_str.size() == 0)
        {
            std::cout << "it must be in form of \"rand(n,min,max)\"" << std::endl;
            return false;
        }
        rand_param_str = rand_param_str.substr(0, rand_param_str.find_last_of(")]"));
        rand_param_str = rand_param_str.substr(0, rand_param_str.find_last_of(")]"));
        rand_param_str = rand_param_str.substr(rand_param_str.find_first_of("([")+1);
        std::vector<std::string> rand_param;
        std::expand::split(rand_param_str, ",; ", rand_param);
        opt.param.resize(rand_param.size());
        for (int i = 0; i < rand_param.size(); ++i)
        {
            opt.param[i] = std::stod(rand_param[i]);
        }
    }
    
    return true;
}

bool aug_png_slover::checkOutOperation_line(const std::string &line, std::vector<___aug_operation> &opt_line)
{
    std::vector<std::string> operation;
    std::expand::split(line, "+", operation);
    for (int i = 0; i < operation.size(); ++i)
    {
        ___aug_operation one_opt;
        if (!checkOutOperation_one(operation[i], one_opt)) return false;
        opt_line.push_back(one_opt);
    }
    return true;
}

bool aug_png_slover::is_set_aug_config()
{
    return is_load_opt;
}

std::vector<std::vector<___aug_operation>> aug_png_slover::getOperation()
{
    return operations;
}

void aug_png_slover::printOperation()
{
    return printOperation(operations);
}

bool aug_png_slover::save_aug_config(const std::vector<std::vector<___aug_operation>> &aug_opts,const std::string &save_path)
{
    std::string config_name;
    std::expand::get_path_last_name(aug_config_file_path, config_name);
    if (config_name.size() == 0) { config_name = "AUG_CONFIG"; }
    
    std::ofstream aug_config_file(save_path + "\\" + config_name);
    if (!aug_config_file.is_open()) return false;
    for (int i = 0; i < aug_opts.size(); ++i)
    {
        aug_config_file << "step" << std::to_string(i + 1) << "=";
        for (int j = 0; j < aug_opts[i].size(); ++j)
        {
            if(j!=0) aug_config_file << "+";
            const std::string &opt_name = aug_opts[i][j].opt_name;
            aug_config_file << opt_name;
            if (opt_name == "FlipUD" || opt_name == "FlipLR" || opt_name == "Flip")
            {
                continue;
            }

            else if (opt_name == "NotBackup")
            {
                continue;
            }
            else if (opt_name == "Pyramid")
            {
                aug_config_file << "[" << aug_opts[i][j].param[0] << " " << aug_opts[i][j].param[1] << "]";
            }
            else
            {
                if (aug_opts[i][j].is_random_param)
                {
                    aug_config_file << "["
                        << "rand("
                        << aug_opts[i][j].param[0] << " " << aug_opts[i][j].param[1] << " " << aug_opts[i][j].param[2]
                        << ")]";
                }
                else
                {
                    aug_config_file << "[";
                    for (int k = 0; k < aug_opts[i][j].param.size(); ++k)
                    {
                        if (k != 0) aug_config_file << " ";
                        aug_config_file << aug_opts[i][j].param[k];
                    }
                    aug_config_file << "]";
                }
            }
        }
        aug_config_file << "          #" << get_step_aug_number_one_mat(aug_opts[i]) << std::endl;
    } 
    aug_config_file.close();
    aug_config_file.clear();
    return true;

}

bool aug_png_slover::save_aug_config(const std::string &save_path)
{
    return save_aug_config(operations, save_path);
}

void aug_png_slover::printOperation(const std::vector<std::vector<___aug_operation>> &aug_opts)
{
    for (int i = 0; i < aug_opts.size(); ++i)
    {
        std::cout << "**************** step" << std::to_string(i + 1) << " ***************" << std::endl;
        const std::vector<___aug_operation> &one_line_operation = aug_opts[i];
        for (int j = 0; j < one_line_operation.size(); ++j)
        {
            std::cout << one_line_operation[j].opt_name;
            //if (one_line_operation[j].fun_type != -1)
            //    std::cout << " tyep = " << one_line_operation[j].fun_type << "    ";
            if(one_line_operation[j].is_random_param) std::cout << "    rand";
            std::cout << std::endl;
            std::expand::print_row(one_line_operation[j].param, 1, 9);
            std::cout << std::endl;
        }
    }
    std::cout << "************** finish ***************" << std::endl;
    return;
}

//bool aug_png_slover::checkOutType_one(___aug_operation &opt)
//{
//    if (opt.opt_name == "FlipUD")
//    {
//        opt.opt_name = "Flip";
//        opt.fun_type  = UD;
//    }
//    else if (opt.opt_name == "FlipLR")
//    {
//        opt.opt_name = "Flip";
//        opt.fun_type = LR;
//    }
//    else if (opt.opt_name == "AffineX")
//    {
//        opt.opt_name = "Affine";
//        opt.fun_type = X;
//    }
//    else if (opt.opt_name == "AffineY")
//    {
//        opt.opt_name = "Affine";
//        opt.fun_type = Y;
//    }
//    else if (opt.opt_name == "PerspectiveUL")
//    {
//        opt.opt_name = "Perspective";
//        opt.fun_type = UL;
//    }
//    else if (opt.opt_name == "PerspectiveU")
//    {
//        opt.opt_name = "Perspective";
//        opt.fun_type = U;
//    }
//    else if (opt.opt_name == "PerspectiveUR")
//    {
//        opt.opt_name = "Perspective";
//        opt.fun_type = UR;
//    }
//    else if (opt.opt_name == "PerspectiveL")
//    {
//        opt.opt_name = "Perspective";
//        opt.fun_type = L;
//    }
//    else if (opt.opt_name == "PerspectiveR")
//    {
//        opt.opt_name = "Perspective";
//        opt.fun_type = R;
//    }
//    else if (opt.opt_name == "PerspectiveDL")
//    {
//        opt.opt_name = "Perspective";
//        opt.fun_type = DL;
//    }
//    else if (opt.opt_name == "PerspectiveD")
//    {
//        opt.opt_name = "Perspective";
//        opt.fun_type = D;
//    }
//    else if (opt.opt_name == "PerspectiveDR")
//    {
//        opt.opt_name = "Perspective";
//        opt.fun_type = DR;
//    }
//    else if(opt.opt_name == "Rotate"
//        || opt.opt_name == "Flip"
//        || opt.opt_name == "Crop"
//        || opt.opt_name == "Affine"
//        || opt.opt_name == "Noise"
//        || opt.opt_name == "Hue"
//        || opt.opt_name == "Saturation"
//        || opt.opt_name == "Lightness"
//        || opt.opt_name == "Perspective"
//        || opt.opt_name == "Pyramid"
//        || opt.opt_name == "NotBackup")
//    {
//        opt.fun_type = -1;
//    }
//    else
//    {
//        std::cout << "Operation error with \"" << opt.opt_name << "\"" << std::endl;
//        return false;
//    }
//    return true;
//}

bool aug_png_slover::checkParam(const std::vector<double> &param, double allow_min, double allow_max)
{
    if (param.size() ==0)
    {
        std::cout << "param error. NULL param" << std::endl;
        return false;
    }
    for (int i = 0; i < param.size(); ++i)
    {
        if (allow_min > param[i] || param[i] > allow_max)
        {
            std::cout << "param error. out of range" << std::endl;
            return false;
        }
    }
    return true;
}

bool aug_png_slover::checkRandomParam(const std::vector<double> &param, double allow_min, double allow_max)
{
    if (param.size() != 3)
    {
        std::cout << "rand param error. n < 0 " << std::endl;
        return false;
    }
    if (param[1] > param[2])
    {
        std::cout << "rand param error. min > max " << std::endl;
        return false;
    }
    if (param[1] < allow_min || allow_max < param[2])
    {
        std::cout << "rand param error. min or max out of range" << std::endl;
        return false;
    }
    return true;
}

int aug_png_slover::get_step_aug_number_one_mat(const std::vector<___aug_operation > &aug_opts)
{
    int n = 0;
    bool no_NotBackup = true;
    bool no_Pyramid = true;
    for (int j = 0; j < aug_opts.size(); ++j)
    {
        const ___aug_operation &opt = aug_opts[j];
        if (opt.opt_name == "Rotate"|| opt.opt_name == "Crop" || opt.opt_name == "Noise"||opt.opt_name=="Distort"
            || opt.opt_name == "Affine" || opt.opt_name == "AffineX" || opt.opt_name == "AffineY"
            || opt.opt_name == "Hue"|| opt.opt_name == "Saturation"|| opt.opt_name == "Lightness"
            || opt.opt_name == "PerspectiveUL" || opt.opt_name == "PerspectiveU" || opt.opt_name == "PerspectiveUR"
            || opt.opt_name == "PerspectiveL" || opt.opt_name == "Perspective" || opt.opt_name == "PerspectiveR"
            || opt.opt_name == "PerspectiveDL" || opt.opt_name == "PerspectiveD" || opt.opt_name == "PerspectiveDR")
        {
            if (opt.is_random_param)
            {
                n += opt.param[0];
            }
            else
            {
                n += opt.param.size();
            }
        }
        else if (opt.opt_name == "Flip" || opt.opt_name == "FlipUD" || opt.opt_name == "FlipLR")
        {
            ++n;
        }
        else if (opt.opt_name == "Pyramid")
        {
            n += opt.param[0];
            ++n;
            n += opt.param[1];
            no_Pyramid = false;
        }
        else if (opt.opt_name == "NotBackup")
        {
            no_NotBackup = false;
        }
    }
    if (no_NotBackup && no_Pyramid)
    {
        ++n;
    }
    return n;
}

bool aug_png_slover::checkOperation(const std::vector<std::vector<___aug_operation>> &aug_opts)
{
    for (int i = 0; i < aug_opts.size(); ++i)
    {
        for (int j = 0; j < aug_opts[i].size(); ++j)
        {
            const ___aug_operation &opt = aug_opts[i][j];
            if (opt.opt_name == "Rotate")
            {
                if (opt.param.size() == 0)
                {
                    std::cout << opt.opt_name << " param error." << std::endl;
                    return false;
                }
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, -100000, 100000))
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }
            else if (opt.opt_name == "Flip"|| opt.opt_name == "FlipUD"|| opt.opt_name == "FlipLR")
            {
            }
            else if (opt.opt_name == "Crop")
            {
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, 0.01, 0.99))
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
                else
                {
                    if (!checkParam(opt.param, 0.1, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }
            else if (opt.opt_name == "Affine"|| opt.opt_name == "AffineX" || opt.opt_name == "AffineY")
            {
                if (opt.param.size() == 0)
                {
                    std::cout << opt.opt_name << " param error." << std::endl;
                    return false;
                }
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, -100000, 100000)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }
            else if (opt.opt_name == "Noise")
            {
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, 0.01, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
                else
                {
                    if (!checkParam(opt.param, 0.01, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }
            else if (opt.opt_name == "Hue")
            {
                if (opt.param.size() == 0)
                {
                    std::cout << opt.opt_name << " param error." << std::endl;
                    return false;
                }
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, -100000, 100000)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }

            else if (opt.opt_name == "Saturation")
            {
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, -0.99, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
                else
                {
                    if (!checkParam(opt.param, -0.99, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }
            else if (opt.opt_name == "Lightness")
            {
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, -0.99, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
                else
                {
                    if (!checkParam(opt.param, -0.99, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }
            else if (opt.opt_name == "PerspectiveUL"|| opt.opt_name == "PerspectiveU"|| opt.opt_name == "PerspectiveUR"
                || opt.opt_name == "PerspectiveL" || opt.opt_name == "Perspective" || opt.opt_name == "PerspectiveR"
                || opt.opt_name == "PerspectiveDL" || opt.opt_name == "PerspectiveD" || opt.opt_name == "PerspectiveDR")
            {
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, 0.01, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
                else
                {
                    if (!checkParam(opt.param, 0.01, 0.99)) 
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }
            else if (opt.opt_name == "Distort")
            {
                if (opt.is_random_param)
                {
                    if (!checkRandomParam(opt.param, -100000, 100000))
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
                else
                {
                    if (opt.param.size() == 0)
                    {
                        std::cout << opt.opt_name << " param error." << std::endl;
                        return false;
                    }
                }
            }
            else if (opt.opt_name == "Pyramid")
            {
                if (opt.param.size() != 2)
                {
                    std::cout << opt.opt_name << " param error." << std::endl;
                }
                else if(aug_opts[i].size()!=1)
                {
                    std::cout <<"Should not have others operation in a Pyramid step." << std::endl;
                }
            }
            else if (opt.opt_name == "NotBackup")
            {
            }
            else
            {
                std::cout << "Operation error with \"" << opt.opt_name << "\"" << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool aug_png_slover::checkOutPreStepList(int step_number, const std::string &save_root_path, std::vector<std::vector<std::string>> &list)
{
    std::vector<std::vector<std::string>> col_first_list;
    if (!std::expand::check_out_list(save_root_path+"\\step"+ std::to_string(step_number-1) + "\\list.txt", " ", list))
    {
        std::cout << "Can not check out pre step list." << std::endl;
        return false;
    }
    return true;
}

int aug_png_slover::check_continue_step(const std::string &save_root_path)
{
    int step_number = 0;
    while (std::expand::is_directory(save_root_path + "\\step" + std::to_string(step_number)))
    {
        step_number++;
    }
    if (step_number != 0) --step_number;
    return step_number;

}

//bool aug_png_slover::check_have_mask(int step_number,const std::string &save_root_path)
//{
//    return fs::exists(save_root_path + "\\step" + std::to_string(step_number) + "\\mask");
//}

void aug_png_slover::get_rand(int n, double min, double max, std::vector<double> &output)
{
    if (n <= 0) return;
    static std::default_random_engine e(time(0));
    std::uniform_real_distribution<double> dist(min, max);

    output.resize(n);
    for (int i = 0; i < n; ++i)
    {
        output[i] = dist(e);
    }
    return;
}

void aug_png_slover::get_rand(const std::vector<double> n_min_max, std::vector<double> &output)
{
    return get_rand(n_min_max[0], n_min_max[1], n_min_max[2], output);
}

bool aug_png_slover::augment_step(int step_number, const  std::string &save_root_path)
{
    std::string step_path = save_root_path + "\\step" + std::to_string(step_number);
    std::string pre_step_path = save_root_path + "\\step" + std::to_string(step_number - 1);

    std::expand::create_directory(step_path);

    std::ofstream list_file(step_path + "\\list.txt");
    if (!list_file.is_open()) return false;

    std::vector<___aug_operation> &step_opt = operations[step_number - 1];
    //std::vector<std::thread * > th(thread_number);
    int pre_save_first_index = 1;
    int aug_number_one_mat = get_step_aug_number_one_mat(step_opt);
    int save_first_index = 1;

    now_step_obj.resize(0);
    now_step_label.resize(0);

    std::vector<cv::Mat> aug_obj_one_mat;
    //int rest_begin = pre_step_obj.size() / thread_number * thread_number;
    int rest_begin = 0;
    for (int j = rest_begin; j < pre_step_obj.size(); ++j)
    {
        augment_one_mat_obj(pre_step_obj[j], aug_obj_one_mat, step_opt);
        for (int i = 0; i < aug_obj_one_mat.size(); ++i)
        {
            //save obj
            cv::imwrite(step_path + "\\obj_" + std::to_string(i + save_first_index) + ".png", aug_obj_one_mat[i]);
            now_step_obj.push_back(step_path + "\\obj_" + std::to_string(i + save_first_index) + ".png");
            //save list
            list_file << step_path + "\\obj_" + std::to_string(i + save_first_index) + ".png";
            if (pre_step_label[j].size() != 0)
            {
                now_step_label.push_back(pre_step_label[j]);
                list_file << " " << pre_step_label[j];
            }
            list_file << std::endl;
        }
        save_first_index += aug_number_one_mat;
    }

    
    pre_step_obj.assign  (now_step_obj  .begin(), now_step_obj.end());
    pre_step_label.assign(now_step_label.begin(), now_step_label.end());

    std::cout << now_step_label[0]<<" save " << now_step_obj .size() << " in step " << step_number << "." << std::endl;

    list_file.close();
    list_file.clear();
    return true;
}

bool aug_png_slover::augment_continue(const std::string &save_root_path)
{
    std::vector<std::string> file_name_list;
    std::expand::scan_one_path_file(save_root_path,".*", file_name_list,std::expand::SCAN_ONLY_FILE_FULL_NAME);

    for (int i = 0; i < file_name_list.size(); ++i)
    {
        if (set_aug_config(save_root_path + '\\' + file_name_list[i])) { break; }
    }

    if (!is_load_opt)
    {
        std::cout << "No load operations." << std::endl;
        return false;
    }

    std::vector<std::string> folder_name_list;
    std::expand::scan_one_path_directory(save_root_path, folder_name_list);
    std::vector<bool> is_folder_4_aug(folder_name_list.size(), false);


    std::vector<int> step_rate;
    step_rate.push_back(1);
    for (int i = 0; i < operations.size(); ++i)
    {
        step_rate.push_back(get_step_aug_number_one_mat(operations[i])*step_rate[step_rate.size() - 1]);
    }

    //step;
    for (int m = 0; m < folder_name_list.size(); ++m)
    {
        std::vector<std::vector<std::string>> list;
        //step0 list
        if (!checkOutPreStepList(1, save_root_path + "\\" + folder_name_list[m], list)) { continue; }
        int step0_number = list[0].size();
        int continue_step = 1;
        //stepi list
        for (int i = 1; i < operations.size() + 1; ++i)
        {
            pre_step_obj = list[0];
            pre_step_label = list[1];
            if (!checkOutPreStepList(i + 1, save_root_path + "\\" + folder_name_list[m], list))
            {
                continue_step = i;
                break;
            }
            else if(list[0].size()!= step0_number*step_rate[i])
            {
                continue_step = i;
                break;
            }
        }

        is_folder_4_aug[m] = true;

        for (int i = continue_step - 1; i < operations.size(); ++i)
        {
            if (!augment_step(i + 1, save_root_path + "\\" + folder_name_list[m]))
            {
                std::cout << folder_name_list[m] << " Fail in " << i + 1 << "step" << std::endl;
                return false;;
            }
        }
    }

    //save all list
    std::ofstream all_list_file(save_root_path + "\\list.txt");
    for (int m = 0; m < folder_name_list.size(); ++m)
    {
        if (is_folder_4_aug[m])
        {
            std::vector<std::vector<std::string>> all_list;
            std::string obj_list_file_path = save_root_path + "\\" + folder_name_list[m] + "\\step" + std::to_string(operations.size()) + "\\list.txt";
            std::ifstream obj_list_file(obj_list_file_path);
            while (!obj_list_file.eof())
            {
                std::string line;
                std::getline(obj_list_file, line);
                if (line.size() != 0) { all_list_file << line << std::endl; }
            }
            obj_list_file.close();
            obj_list_file.clear();
        }
    }
    all_list_file.close();
    all_list_file.clear();

    return true;
}

bool aug_png_slover::augment(const std::vector<std::vector<std::string>> &obj_path, const std::vector<std::string> &label, const  std::string &save_root_path)
{
    if (!is_load_opt)
    {
        std::cout << "No load operations." << std::endl;
        return false;
    }
    std::expand::create_directory(save_root_path);
    for (int m = 0; m < label.size(); ++m)
    {
        std::expand::create_directory(save_root_path + "\\" + label[m]);
    }
    save_aug_config(save_root_path);

    //make mask config
    mask_maker mm(save_root_path + "\\MASK_CONFIG");
    std::vector<mask_config> loaded_config;
    std::vector<std::vector<mask_config>> matting_config;
    if (!mm.load_config(save_root_path + "\\MASK_CONFIG", loaded_config)) { loaded_config.resize(0); }
    if (!mm.make_config(obj_path, matting_config, loaded_config)) { return false; }
    mm.save_config(matting_config);
    
    //step0;
    for (int m = 0; m < label.size(); ++m)
    {
        std::string step_0_save_path = save_root_path + "\\" + label[m] + "\\step0";
        std::expand::create_directory(step_0_save_path);
        std::ofstream file(step_0_save_path + "\\list.txt");
        for (int i = 0; i < obj_path[m].size(); ++i)
        {
            cv::Mat obj_png = mm.make_mask(matting_config[m][i]);
            if (obj_png.empty()) continue;
            obj_png = mask_maker::compactPNG(obj_png);
            if (obj_png.empty()) continue;

            std::string obj_name;
            std::expand::get_path_last_name(obj_path[m][i], obj_name);

            size_t index = obj_name.find_last_of(".");
            std::string front_name = obj_name.substr(0, index);
            std::string extend_name = obj_name.substr(index + 1);


            cv::imwrite(step_0_save_path + "\\" + front_name+".png", obj_png);
            file << step_0_save_path + "\\" + front_name + ".png";
            file << " ";
            file << label[m];
            file << std::endl;
        }
    }

    //step others;
    for (int m = 0; m < label.size(); ++m)
    {
        std::vector<std::vector<std::string>> list;
        checkOutPreStepList(1, save_root_path + "\\" + label[m], list);
        pre_step_obj = list[0];
        pre_step_label = list[1];
        for (int i = 0; i < operations.size(); ++i)
        {
            if (!augment_step(i + 1, save_root_path+"\\"+label[m]))
            {
                std::cout << label[m] << " Fail in "<< i + 1 << "step" << std::endl;
                return false;;
            }
        }
    }

    //save all list
    std::ofstream all_list_file(save_root_path + "\\list.txt");
    for (int m = 0; m < label.size(); ++m)
    {
        std::vector<std::vector<std::string>> all_list;
        std::string obj_list_file_path = save_root_path + "\\" + label[m] + "\\step" + std::to_string(operations.size()) + "\\list.txt";
        std::ifstream obj_list_file(obj_list_file_path);
        while (!obj_list_file.eof())
        {
            std::string line;
            std::getline(obj_list_file, line);
            if (line.size() != 0) { all_list_file << line << std::endl; }
        }
        obj_list_file.close();
        obj_list_file.clear();
    }
    all_list_file.close();
    all_list_file.clear();

    return true;
}
//
//bool aug_png_slover::augment(const std::vector<std::string> &obj_path, const  std::string &save_root_path)
//{
//    if (!is_load_opt)
//    {
//        std::cout << "No load operations." << std::endl;
//        return false;
//    }
//    save_aug_config(save_root_path);
//    //step0;
//    _mkdir((save_root_path + "\\step0").c_str());
//    _mkdir((save_root_path + "\\step0\\obj").c_str());
//    std::ofstream list_file(save_root_path + "\\step0\\list.txt");
//    if (!list_file.is_open()) return false;
//    for (int i = 0; i < obj_path.size(); ++i)
//    {
//        cv::imwrite(save_root_path + "\\step0\\obj\\obj_" + std::to_string(i + 1) + ".jpg", cv::imread(obj_path[i]));
//        list_file << save_root_path + "\\step0\\obj\\obj_" + std::to_string(i + 1) + ".jpg";
//        list_file << std::endl;
//    }
//    pre_step_obj = obj_path;
//    pre_step_label.resize(0);
//
//    //step others;
//    for (int i = 0; i < operations.size(); ++i)
//    {
//        if (!augment_step(i + 1, save_root_path))
//        {
//            std::cout << "Fail in " << i + 1 << "step" << std::endl;
//            return false;;
//        }
//    }
//    return true;
//}
