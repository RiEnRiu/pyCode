#include"string_expand.h"
#include<boost/algorithm/string.hpp>


namespace std
{
    namespace expand
    {
        void find_all(const string &input, const string &pattern, vector<size_t> &results)
        {
            results.resize(0);
            if (input.size() == 0) { return; }
            size_t wait_find_first_index = 0;
            size_t find_index;
            while (1)
            {
                find_index = input.substr(wait_find_first_index).find(pattern);
                if (find_index == string::npos) { return; }
                wait_find_first_index += find_index;
                results.push_back(wait_find_first_index);
                ++wait_find_first_index;
            }
        }

        void find_all_first_of(const string &input, const string &pattern, vector<size_t> &results)
        {
            results.resize(0);
            if (input.size() == 0) { return; }
            size_t wait_find_first_index = 0;
            size_t find_index;
            while (1)
            {
                find_index = input.substr(wait_find_first_index).find_first_of(pattern);
                if (find_index == string::npos) { return; }
                wait_find_first_index += find_index;
                results.push_back(wait_find_first_index);
                ++wait_find_first_index;
            }
        }

        void eliminate(const string &input, const string &pattern, string &results)
        {
            results.resize(0);
            for (int i = 0; i < input.size(); ++i)
            {
                if (pattern.find_first_of(input[i]) == string::npos)
                {
                    results.push_back(input[i]);
                }
            }
            return;
        }
        
        void split(const string &input, const string &pattern, vector<string> &results)
        {
            boost::split(results, input, boost::is_any_of(pattern), boost::token_compress_on);
            if (results.size() > 0)
            {
                if (results[results.size() - 1].size() == 0) { results.pop_back(); }
            }
            if (results.size() > 0)
            {
                

                if (results[0].size() == 0) 
                { 
                    for (int i = 0; i < results.size() - 1; ++i)
                    {
                        results[i] = results[i + 1];
                    }
                    results.pop_back();
                }
            }
            return;
        }

        void split2(const string input, const string &pattern, std::string &part1, std::string &part2, int from_which)
        {
            if (input.size() == 0) { part1 = ""; part2 = ""; return; }
            if (from_which == 0) { return; }
            else if (from_which > 0)
            {
                vector<int> input_flag(input.size(), 0);
                int pre_part_index = 0;
                int part_index = 0;
                for (size_t i = 0; i < input.size() && part_index <= from_which; ++i)
                {
                    bool is_not_pattern = pattern.find_last_of(input[i]) == string::npos;
                    if (is_not_pattern)
                    {
                        input_flag[i] = 0;
                    }
                    else
                    {
                        if (pre_part_index == 0)
                        {
                            part_index += 1;
                            input_flag[i] = part_index;
                        }
                        else
                        {
                            input_flag[i] = input_flag[i - 1];
                        }
                    }
                    pre_part_index = input_flag[i];
                }

                size_t part1_end = input.size();
                size_t part2_begin = input.size();
                size_t ___i = 0;
                for (; ___i < input.size(); ++___i)
                {
                    if (input_flag[___i] == from_which) { part1_end = ___i; break; }
                }
                for (; ___i < input.size(); ++___i)
                {
                    if (input_flag[___i] != from_which) { part2_begin = ___i; break; }
                }
                part1 = input.substr(0, part1_end);
                part2 = input.substr(part2_begin);
                return;
            }
            else
            {
                from_which = -from_which;
                vector<int> input_flag(input.size(), 0);
                int pre_part_index = 0;
                int part_index = 0;
                for (int i = input.size() - 1; i >= 0 && part_index <= from_which; --i)
                {
                    bool is_not_pattern = pattern.find_last_of(input[i]) == string::npos;
                    if (is_not_pattern)
                    {
                        input_flag[i] = 0;
                    }
                    else
                    {
                        if (pre_part_index == 0)
                        {
                            part_index += 1;
                            input_flag[i] = part_index;
                        }
                        else
                        {
                            input_flag[i] = input_flag[i + 1];
                        }
                    }
                    pre_part_index = input_flag[i];
                }

                int part1_end = 0;
                int part2_begin = 0;
                int ___i = input.size() - 1;
                for (; ___i >= 0; --___i)
                {
                    if (input_flag[___i] == from_which) { part2_begin = ___i + 1; break; }
                }
                for (; ___i >= 0; --___i)
                {
                    if (input_flag[___i] != from_which) { part1_end = ___i + 1; break; }
                }
                part1 = input.substr(0, part1_end);
                part2 = input.substr(part2_begin);
                return;
            }
        }

        void upper_letter(string &inputOutputStr)
        {
            for (int i = 0; i < inputOutputStr.size(); ++i)
            {
                if (97 <= inputOutputStr[i] && inputOutputStr[i] <= 122)
                {
                    inputOutputStr[i] -= 32;
                }
            }
            return;
        }

        void lower_letter(string &inputOutputStr)
        {
            for (int i = 0; i < inputOutputStr.size(); ++i)
            {
                if (65 <= inputOutputStr[i] && inputOutputStr[i] <= 90)
                {
                    inputOutputStr[i] += 32;
                }
            }
            return;
        }

        bool is_letter_equal(const string &input1, const string &input2)
        {
            if (input1.size() != input2.size()) { return false; }
            for (int i = 0; i < input1.size(); ++i)
            {
                if (input1[i] != input2[i])
                {
                    if (65 <= input1[i] && input1[i] <= 90)
                    {
                        if (input1[i] + 32 != input2[i]) { return false; }
                    }
                    else if (97 <= input1[i] && input1[i] <= 122)
                    {
                        if (input1[i] - 32 != input2[i]) { return false; }
                    }
                }
            }
            return true;
        }
    }
}
