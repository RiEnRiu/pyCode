#include"file_expand.h"
#include"vector_expand.hpp"
#include"string_expand.h"
#include<boost/filesystem.hpp>


//#include<fstream>
//#include<vector>
//#include<io.h>
//#include<direct.h>

namespace bfs = boost::filesystem;

namespace std
{
    namespace expand
    {
        void system_complete(const string &input, string &output)
        {
            output = bfs::system_complete(input).string();
            return;
        }

        void get_path_last_name(const string &path, string &name)
        {
            name = bfs::path(path).filename().string();
            return;
        }

        bool is_absolute(const string &path)
        {
            return bfs::path(path).is_absolute();
        }

        bool is_relative(const string &path)
        {
            return bfs::path(path).is_relative();
        }

        bool get_parent_absolute_path(const string &path, string &parent_path)
        {
            auto _path = bfs::path(path);
            if (!bfs::is_directory(_path)) { parent_path = ""; return false; }
            auto t_path = bfs::system_complete(_path);
            parent_path = t_path.parent_path().string();
            return true;
        }

        void get_cwd(string &path)
        {
            path = bfs::current_path().string();
            return;
        }

        bool is_regular_file(const string &path_fname)
        {
            return bfs::is_regular_file(path_fname);
        }

        bool is_directory(const string &path)
        {
            return bfs::is_directory(path);
        }

        bool scan_one_path_file(const string &path, const string &accepted_expand_name,
            vector<string> & file_name, int mode)
        {
            file_name.resize(0);
            bfs::path one_path(path);
            if (!bfs::is_directory(one_path)) { return false; }


            switch (mode)
            {
            case SCAN_ONLY_FILE_FRONT_NAME:
            {
                if (accepted_expand_name == ".*")
                {
                    bfs::directory_iterator end_iter;
                    for (bfs::directory_iterator iter(one_path); iter != end_iter; ++iter)
                    {
                        if (bfs::is_regular_file(iter->path()))
                        {
                            string one_name = iter->path().filename().string();
                            size_t name_size = one_name.find_last_of('.');
                            if (name_size != string::npos) { file_name.push_back(one_name.substr(0, name_size)); }
                            else { file_name.push_back(one_name); }
                        }
                    }
                }
                else
                {
                    vector<size_t>  index;
                    find_all_first_of(accepted_expand_name, ".", index);
                    vector<string> acc_exn;
                    for (size_t i = 0; i < index.size() - 1; ++i)
                    {
                        size_t one_size = index[i + 1] - index[i];
                        if (one_size > 0) { acc_exn.push_back(accepted_expand_name.substr(index[i], one_size)); }
                    }
                    acc_exn.push_back(accepted_expand_name.substr(index[index.size() - 1]));
                    bfs::directory_iterator end_iter;
                    for (bfs::directory_iterator iter(one_path); iter != end_iter; ++iter)
                    {
                        if (bfs::is_regular_file(iter->path()))
                        {
                            for (int i = 0; i < acc_exn.size(); ++i)
                            {
                                if (is_letter_equal(iter->path().extension().string(), acc_exn[i]))
                                {
                                    string one_name = iter->path().filename().string();
                                    file_name.push_back(one_name.substr(0, one_name.find_last_of('.')));
                                    break;
                                }
                            }
                        }
                    }
                }
                break;
            }
            default:
            {
                if (accepted_expand_name == ".*")
                {
                    bfs::directory_iterator end_iter;
                    for (bfs::directory_iterator iter(one_path); iter != end_iter; ++iter)
                    {
                        if (bfs::is_regular_file(iter->path()))
                        {
                            file_name.push_back(iter->path().filename().string());
                        }
                    }
                }
                else
                {
                    vector<size_t>  index;
                    find_all_first_of(accepted_expand_name, ".", index);
                    vector<string> acc_exn;
                    for (size_t i = 0; i < index.size() - 1; ++i)
                    {
                        size_t one_size = index[i + 1] - index[i];
                        if (one_size > 0) { acc_exn.push_back(accepted_expand_name.substr(index[i], one_size)); }
                    }
                    acc_exn.push_back(accepted_expand_name.substr(index[index.size() - 1]));
                    bfs::directory_iterator end_iter;
                    for (bfs::directory_iterator iter(one_path); iter != end_iter; ++iter)
                    {
                        if (bfs::is_regular_file(iter->path()))
                        {
                            for (int i = 0; i < acc_exn.size(); ++i)
                            {
                                if (is_letter_equal(iter->path().extension().string(), acc_exn[i]))
                                {
                                    file_name.push_back(iter->path().filename().string());
                                    break;
                                }
                            }
                        }
                    }
                }
                break;
            }
            }
            return true;
        }

        bool scan_one_path_directory(const string &path, vector<string> & directory_name, int mode)
        {
            directory_name.resize(0);
            bfs::path one_path(path);
            if (!bfs::is_directory(one_path)) { return false; }

            switch (mode)
            {
            case SCAN_RELATIVE_PATH_FILE_FULL_NAME:
            {
                bfs::directory_iterator end_iter;
                for (bfs::directory_iterator iter(one_path); iter != end_iter; ++iter)
                {
                    if (bfs::is_directory(iter->path()))
                    {
                        directory_name.push_back(iter->path().filename().string());
                    }
                }
                break;
            }
            default:
            {
                bfs::directory_iterator end_iter;
                for (bfs::directory_iterator iter(one_path); iter != end_iter; ++iter)
                {
                    if (bfs::is_directory(iter->path()))
                    {
                        directory_name.push_back(iter->path().string());
                    }
                }
                break;
            }
            }
            return true;
        }

        bool scan_file_list(const string &root_path, const string &accepted_expand_name,
            vector<string> & file_list, int mode, vector<vector<string>> &temp)
        {
            if (!is_directory(root_path)) { return false; }
            temp.resize(3);
            temp[0].resize(0);
            temp[1].resize(0);
            temp[2].resize(0);
            return scan_file_list(root_path, accepted_expand_name, file_list, mode, temp[0], temp[1], temp[2]);
        }

        bool scan_file_list(const string &root, const std::string &pathFromRoot, const vector<string> &accepted_expand_name,
            vector<string> & relative_dir, vector<string> &file_front_name, vector<string> &file_expand_name)
        {
            bool flag = true;
            bfs::directory_iterator end_iter;
            for (bfs::directory_iterator iter(root + ((pathFromRoot.size() == 0) ? ("") : ("/" + pathFromRoot))); iter != end_iter; ++iter)
            {
                if (bfs::is_regular_file(iter->path()))
                {
                    string exn = iter->path().extension().string();
                    for (int i = 0; i < accepted_expand_name.size(); ++i)
                    {
                        if (accepted_expand_name[i] == ".*" || is_letter_equal(accepted_expand_name[i], exn))
                        {
                            file_expand_name.push_back(exn);
                            string file_name = iter->path().filename().string();
                            file_name.resize(file_name.size() - exn.size());
                            file_front_name.push_back(file_name);
                            //cout << file_name << endl;
                            relative_dir.push_back(pathFromRoot);
                            break;
                        }
                    }
                }
                else if(bfs::is_directory(iter->path()))
                {
                    string one_path_from_root;
                    if (pathFromRoot.size() != 0)
                    {
                        one_path_from_root = pathFromRoot + "/" + iter->path().filename().string();

                    }
                    else
                    {
                        one_path_from_root = iter->path().filename().string();
                    }
                    flag = flag && scan_file_list(root, one_path_from_root, accepted_expand_name,
                        relative_dir, file_front_name, file_expand_name);
                }
                else
                {
                    return false;
                }
            }
            return flag;
        }


        bool scan_file_list(const string &root_path, const string &accepted_expand_name,
            vector<string> & file_list, int mode, vector<string> & relative_dir, vector<string> &file_front_name, vector<string> &file_expand_name)
        {
            vector<size_t>  index;
            find_all_first_of(accepted_expand_name, ".", index);
            vector<string> acc_exn;
            for (size_t i = 0; i < index.size() - 1; ++i)
            {
                size_t one_size = index[i + 1] - index[i];
                if (one_size > 0) { acc_exn.push_back(accepted_expand_name.substr(index[i], one_size)); }
            }
            acc_exn.push_back(accepted_expand_name.substr(index[index.size() - 1]));

            //cout << "begin scan" << endl;

            bool flag = scan_file_list(root_path, "", acc_exn, relative_dir, file_front_name, file_expand_name);
            if (!flag) { return false; }
            switch (mode)
            {
            case SCAN_ONLY_FILE_FULL_NAME:
            {
                file_list.resize(file_front_name.size());
                for (int i = 0; i < file_front_name.size(); ++i)
                {
                    file_list[i] = file_front_name[i] + file_expand_name[i];
                }
                break;
            }
            case SCAN_ONLY_FILE_FRONT_NAME:
            {
                file_list = file_front_name;
                break;
            }
            case SCAN_RELATIVE_PATH_FILE_FULL_NAME:
            {
                file_list.resize(file_front_name.size());
                for (int i = 0; i < file_front_name.size(); ++i)
                {
                    if (relative_dir[i].size() != 0)
                    {
                        file_list[i] = relative_dir[i] + "/" + file_front_name[i] + file_expand_name[i];
                    }
                    else
                    {
                        file_list[i] = file_front_name[i] + file_expand_name[i];
                    }
                }
                break;
            }
            default://SCAN_ROOT_PATH_FILE_FULL_NAME
                file_list.resize(file_front_name.size());
                for (int i = 0; i < file_front_name.size(); ++i)
                {
                    if (relative_dir[i].size() != 0)
                    {
                        file_list[i] = root_path + "/" + relative_dir[i] + "/" + file_front_name[i] + file_expand_name[i];
                    }
                    else
                    {
                        file_list[i] = root_path + "/" + file_front_name[i] + file_expand_name[i];
                    }
                }
                break;
            }
            return true;
        }

        bool check_out_list(const string &path_fname, const string &pattern, vector<vector<string>> &list, bool is_col_first)
        {
            list.resize(0);
            ifstream file(path_fname);
            if (!file.is_open())
            {
                return false;
            }

            while (!file.eof())
            {
                std::string line;
                std::getline(file, line);
                if (line.size() == 0) { continue; }
                list.push_back(vector<string>());
                split(line, " ", list[list.size() - 1]);
                if (list[list.size() - 1].size() == 0) { list.pop_back(); }
            }
            file.close();
            file.clear();
            if (is_col_first)
            {
                std::expand::transpose(list, list);
            }
            return true;
        }

        bool create_directory(const string &path)
        {
            if (bfs::is_directory(path)) { return true; }
            else { return bfs::create_directories(path); }
        }

    }
}
