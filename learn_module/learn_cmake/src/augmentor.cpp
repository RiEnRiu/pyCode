// Augmentor.cpp : 定义控制台应用程序的入口点。
//
#include"aug_solver/aug_solver.h"
#include"stl_expand/stl_expand.hpp"

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<vector>
#include<ctime>

#include <boost/program_options.hpp> 
namespace po = boost::program_options;

using namespace std;
using namespace cv;


bool get_obj_label_from_dir(const std::string& dir, std::vector<vector<std::string>>& obj, std::vector<std::string>& label)
{
    vector<string> folder;

    if (!std::expand::scan_one_path_directory(dir, folder)) { return false; }
    obj.resize(folder.size());
    for (int i = 0; i < folder.size(); ++i)
    {
        std::expand::scan_one_path_file(dir + "\\" + folder[i], ".*", obj[i]);
        for (int j = 0; j < obj[i].size(); ++j)
        {
            obj[i][j] = dir + "\\" + folder[i] + "\\" + obj[i][j];
        }
        if (obj[i].size() != 0) { label.push_back(folder[i]); }
    }
    if (obj.size() == 0) return false;
    return true;
}


int main(int argc, char **argv)
{
    string aug_config_path;
    string save_aug_path;

    po::options_description list_desc("");

    string obj_dir;
    string obj_label;
    po::options_description new_desc("NEW");
    new_desc.add_options()
        ("dir", po::value<string>(&obj_dir), "Object direction.")
        ("aug_config", po::value<string>(&aug_config_path)->default_value("./aug_config/AUG_ALL"), "The aug config file.")
        ("save", po::value<string>(&save_aug_path)->default_value("aug"), "The path for output.");


    string restart_path;
    po::options_description restart_desc("RESTART");
    restart_desc.add_options()
        ("path", po::value<string>(&restart_path), "Restart by appointing an saved path.");
    po::options_description all("Augmentor options");
    all.add(list_desc).add(new_desc).add(restart_desc);

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, all), vm);
        po::notify(vm);
    }
    catch (boost::program_options::error_with_no_option_name &ex)
    {
        cout << ex.what() << endl;
        return -1;
    }

    ///////////////////////////////////
    //test 
    if (argc == 1)
    {
        cout << all << endl;
        return -1;
    }
    //obj_dir = "G:\\ver3.0.2\\obj1";
    //aug_config_path = "G:\\ver3.0.2\\aug_config\\AUG_TEST";
    //save_aug_path = "test_new";
    //////////////////////////////////


    aug_png_slover aug_s(aug_config_path);
    if (restart_path.size() != 0)
    {
        if (!aug_s.augment_continue(restart_path))
        {
            std::cout << "Can not restart the project." << std::endl;
            return -1;
        }
        return 0;
    }

    ////////////////////////////////////////////////////
    if (obj_dir.size() != 0)
    {
        vector<vector<string>> obj;
        vector<string> label;
        if (!get_obj_label_from_dir(obj_dir, obj, label))
        {
            cout << "Can not scan any image." << endl;
            return -1;
        }
        else if (!aug_s.augment(obj, label, save_aug_path))
        {
            cout << "Augment fail." << endl;
            return -1;
        }
        return 0;
    }
    ////////////////////////////////////////////////////

    return 0;
}




