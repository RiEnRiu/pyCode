#ifndef MODIFY_ORDER_BY_WEIGHT_H
#define MODIFY_ORDER_BY_WEIGHT_H

#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <set>
#include <iostream>

#define HAND_ID 0
#define IN_STATE 1
#define OUT_STATE (-1)
#define SELECT 100001
#define CHANGE 100002
#define MISS 100003


using namespace std;

class IN_OUT_INFO
{
public:
    int in_out_track;
    int key;
    int hid;
    vector<int> cls_90;
    vector<float> cls_conf_90;
};


class MOD_INFO
{
public:
    int in_out_track;
    int key;
    int hid;
    int cls;
    float cls_conf;
    int mod_state;

    MOD_INFO(){};

    MOD_INFO(
        int _in_out_track,
        int _key,
        int _hid,
        int _cls,
        float _cls_conf,
        int _mod_state)
    {
        in_out_track = _in_out_track;
        key = _key;
        hid = _hid;
        cls = _cls;
        cls_conf = _cls_conf;
        mod_state = _mod_state;
    };

    MOD_INFO(const MOD_INFO &_mod)
    {
        in_out_track = _mod.in_out_track;
        key = _mod.key;
        hid = _mod.hid;
        cls = _mod.cls;
        cls_conf = _mod.cls_conf;
        mod_state = _mod.mod_state;
    };


};

class ORDER_INFO
{
public:
    vector<int> order;
    vector<MOD_INFO> modification;
    int modify_time;
    vector<float> weight_predict;
    bool is_friendly;
    ORDER_INFO() {};
    ORDER_INFO(const ORDER_INFO &_order)
    {
        order.assign(_order.order.begin(), _order.order.end());
        modification.assign(_order.modification.begin(), _order.modification.end());
        modify_time = _order.modify_time;
        weight_predict.assign(_order.weight_predict.begin(), _order.weight_predict.end());
        is_friendly = _order.is_friendly;
    };

    void copyTo(ORDER_INFO &_order) const
    {
        _order.order.assign(order.begin(), order.end());
        _order.modification.assign(modification.begin(), modification.end());
        _order.modify_time = modify_time;
        _order.weight_predict.assign(weight_predict.begin(), weight_predict.end());
        _order.is_friendly = is_friendly;
        return;
    };

};



void _get_no_change_add_weight(
    const vector<IN_OUT_INFO> &in_out_list,
    const vector<float> &goods_weight,
    vector<float> &no_modification_add_weights)
{
    vector<float> reversed_result;
    float this_add_weight = 0;
    //riter = arrayInt.rbegin(); riter != arrayInt.rend(); riter++
    for (auto iter = in_out_list.begin(); iter != in_out_list.end(); ++iter)
    {
        int cls = iter->cls_90[0];
        int in_out_track = iter->in_out_track;
        if (cls != HAND_ID)
        {
            this_add_weight += in_out_track*goods_weight[cls];
        }
        reversed_result.push_back(this_add_weight);
    }
    reverse(reversed_result.begin(), reversed_result.end());
    return;
}


void _modify_order_one_in_out_multicls(
    const ORDER_INFO &order,
    const IN_OUT_INFO &one_in_out,
    int max_modify_times,
    const vector<float> &goods_weight,
    int in_out_time,
    int max_in_out_time,
    vector<ORDER_INFO> &r_order)
{
    size_t changed_order_begin = r_order.size();

    int in_out_track = one_in_out.in_out_track;
    int key = one_in_out.key;
    int	hid = one_in_out.hid;
    vector<int>	cls_90 = one_in_out.cls_90;
    vector<float> cls_conf_90 = one_in_out.cls_conf_90;

    set<int> maybe_cls;

    //cls_90[0], cls_conf_90[0]
    maybe_cls.insert(cls_90[0]);
    int s_cls = cls_90[0];
    float s_cls_conf = cls_conf_90[0];
    ORDER_INFO tmp_order(order);
    float wp = *(tmp_order.weight_predict.end() - 1);
    if (s_cls != HAND_ID)
    {
        wp += in_out_track * goods_weight[s_cls];
        tmp_order.order[s_cls] -= in_out_track;
        if (tmp_order.order[s_cls] < -0.1)
        {
            tmp_order.is_friendly = false;
        }
    }
    tmp_order.weight_predict.push_back(wp);
    r_order.push_back(tmp_order);
    //if (order.modification.size() >= max_modify_times || in_out_time >= max_in_out_time)
    //{
    //    return;
    //}

    //other maybe cls
    for (int i = 1; i < cls_90.size(); ++i)
    {
        int s_cls = cls_90[i];
        float s_cls_conf = cls_conf_90[i];
        ORDER_INFO tmp_order(order);
        tmp_order.modification.push_back(
            MOD_INFO(in_out_track, key, hid, s_cls, s_cls_conf, SELECT));
        float wp = *(tmp_order.weight_predict.end() - 1);
        if (s_cls != HAND_ID)
        {
            wp += in_out_track * goods_weight[s_cls];
            tmp_order.order[s_cls] -= in_out_track;
            if (tmp_order.order[s_cls] < -0.1)
            {
                tmp_order.is_friendly = false;
            }
        }
        tmp_order.weight_predict.push_back(wp);
        r_order.push_back(tmp_order);
    }
    if (order.modification.size() >= max_modify_times || in_out_time >= max_in_out_time)
    {
        return;
    }

    // change to hand
    if (maybe_cls.find(HAND_ID) == maybe_cls.end())
    {
        ORDER_INFO tmp_order(order);
        tmp_order.modification.push_back(
            MOD_INFO(in_out_track, key, hid, HAND_ID, 0.111111, CHANGE));
        tmp_order.modify_time += 1;
        tmp_order.weight_predict.push_back(*(tmp_order.weight_predict.end() - 1));
        r_order.push_back(tmp_order);
    }
    // change to other goods
    for (int goods_id = HAND_ID+1; goods_id < goods_weight.size(); ++goods_id)
    {
        if (maybe_cls.find(goods_id) != maybe_cls.end())
        {
            continue;
        }
        ORDER_INFO tmp_order(order);
        tmp_order.modification.push_back(
            MOD_INFO(in_out_track, key, hid, goods_id, 0.111111, CHANGE));
        tmp_order.modify_time += 1;
        float wp = *(tmp_order.weight_predict.end() - 1);
        wp += in_out_track * goods_weight[goods_id];
        tmp_order.weight_predict.push_back(wp);
        tmp_order.order[goods_id] -= in_out_track;
        if (tmp_order.order[goods_id] < -0.1)
        {
            tmp_order.is_friendly = false;
        }
        r_order.push_back(tmp_order);
    }
    if (order.modification.size() + 1 >= max_modify_times)
    {
        return;
    }

    // miss one goods
    size_t changed_order_end = r_order.size();
    for (int goods_id = HAND_ID+1; goods_id < goods_weight.size(); ++goods_id)
    {
        ORDER_INFO tmp_order(r_order[changed_order_begin]);
        tmp_order.modification.push_back(
            MOD_INFO(in_out_track, key, hid, goods_id, 0.111111, MISS));
        tmp_order.modify_time += 1;
        float wp = *(tmp_order.weight_predict.end() - 1);
        wp += in_out_track * goods_weight[goods_id];
        tmp_order.weight_predict[tmp_order.weight_predict.size() - 1] = wp;
        tmp_order.order[goods_id] -= in_out_track;
        if (tmp_order.order[goods_id] < -0.1)
        {
            tmp_order.is_friendly = false;
        }
        r_order.push_back(tmp_order);
    }
    //if (order.modification.size() + 1 >= max_modify_times)
    //{
    //    return;
    //}

    //miss one goods
    for (int i = changed_order_begin + 1; i < changed_order_end; ++i)
    {
        for (int goods_id = HAND_ID + 1; goods_id < goods_weight.size(); ++goods_id)
        {
            ORDER_INFO tmp_order(r_order[i]);
            tmp_order.modification.push_back(
                MOD_INFO(in_out_track, key, hid, goods_id, 0.111111, MISS));
            tmp_order.modify_time += 1;
            float wp = *(tmp_order.weight_predict.end() - 1);
            wp += in_out_track * goods_weight[goods_id];
            tmp_order.weight_predict[tmp_order.weight_predict.size() - 1] = wp;
            tmp_order.order[goods_id] -= in_out_track;
            if (tmp_order.order[goods_id] < -0.1)
            {
                tmp_order.is_friendly = false;
            }
            r_order.push_back(tmp_order);
        }
    }
    return;
}


int _weight_match_times(const vector<float> &weight_predict,
    const vector<float> &weight_measure,
    float weight_thresh)
{
    int len1 = weight_predict.size();
    int len2 = weight_measure.size();
    vector<vector<int>> matrix = vector<vector<int>>();
    for (int i = 0; i < len1 + 1; ++i)
    {
        matrix.push_back(vector<int>(len2 + 1, 0));
    }
    for (int i = 0; i < len1; ++i)
    {
        for (int j = 0; j < len2; ++j)
        {
            if (-weight_thresh < weight_predict[i] - weight_measure[j] &&
                weight_predict[i] - weight_measure[j] < weight_thresh)
            {
                matrix[i + 1][j + 1] = matrix[i][j] + 1;
            }
            else
            {
                matrix[i + 1][j + 1] = matrix[i][j + 1] > matrix[i + 1][j] ? matrix[i][j + 1] : matrix[i + 1][j];
            }
        }
    }
    return matrix[len1][len2];
}

void _find_max_wmt(
    const vector<size_t> &index_range,
    const vector<ORDER_INFO> &all_orders,
    const vector<float> &weight_measure,
    float weight_thresh,
    vector<size_t> &r_index)
{
    int max_wmt = -1;
    for (auto iter = index_range.begin(); iter != index_range.end(); ++iter)
    {
        size_t i = *iter;
        //cout << "begin to wmt" << endl;
        int wmt = _weight_match_times(all_orders[i].weight_predict, weight_measure, weight_thresh);
        //cout << "finish wmt" << endl;
        if (max_wmt < wmt)
        {
            {
                vector<size_t> t(1, i);
                r_index.swap(t);
            }
            max_wmt = wmt;
        }
        else if (max_wmt == wmt)
        {
            r_index.push_back(i);
        }
    }
    return;
}

void _split_friendly(
    const vector<size_t> &index_range,
    const vector<ORDER_INFO> &all_orders,
    vector<size_t> &index_friendly,
    vector<size_t> &index_not_friendly)
{
    for (auto iter = index_range.begin(); iter != index_range.end(); ++iter)
    {
        size_t i = *iter;
        if (all_orders[i].is_friendly)
        {
            index_friendly.push_back(i);
        }
        else
        {
            index_not_friendly.push_back(i);
        }
    }
    return;
}

void _find_min_mod_multicls(
    const vector<size_t> &index_range,
    const vector<ORDER_INFO> &all_orders,
    vector<size_t> &r_index)
{
    int min_modify_times = 10000000;
    int min_modification = 10000000;
    for (auto iter = index_range.begin(); iter != index_range.end(); ++iter)
    {
        size_t i = *iter;
        int modify_times = all_orders[i].modify_time;
        int len_modification = all_orders[i].modification.size();
        if (min_modify_times > modify_times)
        {
            {
                vector<size_t> t(1, i);
                r_index.swap(t);
            }
            min_modify_times = modify_times;
            min_modification = len_modification;
        }
        else if (min_modify_times == modify_times)
        {
            if (min_modification >len_modification)
            {
                {
                    vector<size_t> t(1, i);
                    r_index.swap(t);
                }
                min_modification = len_modification;
            }
            else
            {
                r_index.push_back(i);
            }
        }
    }
    return;
}

size_t _find_min_dw(
    const vector<size_t> &index_range,
    const vector<ORDER_INFO> &all_orders,
    const vector<float> &weight_measure)
{
    size_t r_index = index_range[0];
    float min_dw = 100000000.0f;
    for (auto iter = index_range.begin(); iter != index_range.end(); ++iter)
    {
        size_t i = *iter;
        float wp = *(all_orders[i].weight_predict.end() - 1);
        float wm = *(weight_measure.end() - 1);
        float dw = wp - wm;
        dw = dw > 0 ? dw : -dw;
        if ( dw < min_dw)
        {
            min_dw = dw;
            r_index = i;
        }
    }
    return r_index;
}


void _choice_optimum_multicls(
    const vector<ORDER_INFO> &all_orders,
    const vector<float> &weight_measure,
    float weight_thresh,
    ORDER_INFO &opt_order)
{
    //1st.find low weight thresh
    vector<size_t> index_low_weight_thresh;
    vector<size_t> index_over_weight_thresh;
    for (int i = 0; i < all_orders.size(); ++i)
    {
        const ORDER_INFO &order = all_orders[i];
        float wp = *(order.weight_predict.end() - 1);
        float wm = *(weight_measure.end() - 1);
        if (-weight_thresh < wp - wm && wp - wm < weight_thresh)
        {
            //cout << int((wp - wm > 0 ? wp - wm : -wp + wm)*1000) << " * ";
            index_low_weight_thresh.push_back(i);
        }
        else
        {
            index_over_weight_thresh.push_back(i);
        }
    }
    //cout << endl;
    //cout << "weight_thresh = " << weight_thresh<<endl;
    //size_t test_i = all_orders.size() - 1;
    //cout << "weight predict = " << endl;
    //for (int i = 0; i < all_orders[test_i].weight_predict.size(); ++i)
    //{
    //    cout << all_orders[test_i].weight_predict[i] << endl;
    //}
    //cout << "weight measure = " << endl;
    //for (int i = 0; i < weight_measure.size(); ++i)
    //{
    //    cout << weight_measure[i] << endl;
    //}
    //cout << index_low_weight_thresh.size()<<"    "<< index_over_weight_thresh.size() <<endl;


    //2nd. return visual case if all over weight_thresh
    if (index_low_weight_thresh.size() == 0)
    {
        for (auto p_order = all_orders.begin(); p_order != all_orders.end(); ++p_order)
        {
            if (p_order->modification.size() == 0)
            {
                p_order->copyTo(opt_order);
                return;
            }
        }
    }

    //3rd.find max weight match times
    vector<size_t> index_max_wmt;
    _find_max_wmt(index_low_weight_thresh, all_orders,
        weight_measure, weight_thresh,
        index_max_wmt);
    //cout << "index_max_wmt = " << index_max_wmt.size()<< endl;

    //4th.split friendly
    vector<size_t> index_friendly_in_max_wmt;
    vector<size_t> index_not_friendly_in_max_wmt;
    _split_friendly(index_max_wmt, all_orders,
        index_friendly_in_max_wmt, index_not_friendly_in_max_wmt);

    //5th.min modifications
    if (index_friendly_in_max_wmt.size() == 0)
    {
        vector<size_t> index_min_mod_in_not_friendly_in_max_wmt;
        _find_min_mod_multicls(index_not_friendly_in_max_wmt, all_orders, index_min_mod_in_not_friendly_in_max_wmt);
        size_t i = _find_min_dw(index_min_mod_in_not_friendly_in_max_wmt, all_orders, weight_measure);
        all_orders[i].copyTo(opt_order);
        return;
    }
    else
    {
        vector<size_t> index_min_mod_in_friendly_in_max_wmt;
        _find_min_mod_multicls(index_friendly_in_max_wmt, all_orders, index_min_mod_in_friendly_in_max_wmt);
        size_t i = _find_min_dw(index_min_mod_in_friendly_in_max_wmt, all_orders, weight_measure);
        all_orders[i].copyTo(opt_order);
        return;
    }
}

void cpp_modify_order(
    const ORDER_INFO &order,
    const vector<IN_OUT_INFO> &in_out_list,
    const vector<float> &weight_measure,
    const vector<float> &goods_weight,
    const float weight_thresh,
    const int max_modify_times,
    const int max_in_out_time,
    ORDER_INFO &opt_order)
{
    vector<ORDER_INFO> all_orders = vector<ORDER_INFO>();
    all_orders.push_back(order);

    //vector<float> no_modification_add_weights;
    //_get_no_change_add_weight(in_out_list, goods_weight,
    //	no_modification_add_weights);
    int in_out_time = 0;
    for (auto p_one_in_out = in_out_list.begin(); p_one_in_out != in_out_list.end(); ++p_one_in_out, ++in_out_time)
    {
        vector<ORDER_INFO> r_order;
        for (auto p_order = all_orders.begin(); p_order != all_orders.end(); ++p_order)
        {
            _modify_order_one_in_out_multicls(
                *p_order, *p_one_in_out,
                max_modify_times, goods_weight,
                in_out_time, max_in_out_time,
                r_order);
        }
        all_orders.swap(r_order);
        //cout << in_out_time + 1 << " " << all_orders.size() << endl;
    }

    _choice_optimum_multicls(all_orders, weight_measure,
        weight_thresh, opt_order);
    return;
}

#endif // !MODIFY_ORDER_BY_WEIGHT_H
