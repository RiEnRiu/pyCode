#ifndef STRING_EXPAND_H
#define STRING_EXPAND_H

#include<string>
#include<vector>

namespace std
{
    namespace expand
    {
        //*find all index of pattern
        void find_all(const string &input, const string &pattern, vector<size_t> &results);

        //*find_first_of all index of pattern
        void find_all_first_of(const string &input, const string &pattern, vector<size_t> &results);

        //*delete pattern
        void eliminate(const string &input, const string &pattern, string &results);

        //boost::split(results, input, boost::is_any_of(pattern), boost::token_compress_on);
        void split(const string &input, const string &pattern, vector<string> &results);

        //*split into 2 parts by the first of pattern
        void split2( const string input, const string &pattern, std::string &part1, std::string &part2, int from_which = 1);

        //*convert letter
        void upper_letter(string &inputOutputStr);
        void lower_letter(string &inputOutputStr);
        bool is_letter_equal(const string &input1, const string &input2);
    }

}





#endif // 
