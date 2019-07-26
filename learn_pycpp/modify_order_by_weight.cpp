#ifndef MODIFY_ORDER_BY_WEIGHT_CPP
#define MODIFY_ORDER_BY_WEIGHT_CPP

#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <map>
#include <string>
#include "modify_order_by_weight.h"

namespace bp = boost::python;

bp::object modify_order(
    bp::object order, bp::object in_out_list, 
    bp::object weight_measure, bp::object goods_weight, 
    bp::object weight_thresh, bp::object modify_times_thresh,
    bp::object max_in_out_time)
{
    //cpp_goods_weight, id_to_goods, goods_to_id
    map<int, string> id_to_goods;
    map<string, int> goods_to_id;
    goods_to_id["hand"] = HAND_ID;
    id_to_goods[HAND_ID] = "hand";
    bp::dict py_goods_weight = bp::extract<bp::dict>(goods_weight);
    bp::list py_goods_weight_items = py_goods_weight.items();
    vector<float> cpp_goods_weight(bp::len(py_goods_weight_items)+1+ HAND_ID);
    for (int i = 0; i < bp::len(py_goods_weight_items); ++i)
    {
        bp::tuple one_item = bp::extract<bp::tuple>(py_goods_weight_items[i]);
        int cpp_id = i + 1 + HAND_ID;
        string cpp_goods = bp::extract<string>(one_item[0]);
        float cpp_weight = bp::extract<float>(one_item[1]);
        id_to_goods[cpp_id] = cpp_goods;
        goods_to_id[cpp_goods] = cpp_id;
        cpp_goods_weight[cpp_id] = cpp_weight;
    }
    //cout << "id_to_goods" << endl;
    //for (auto iter = id_to_goods.begin(); iter!=id_to_goods.end(); ++iter)
    //{
    //    cout << iter->first << "    "<<iter->second << endl;
    //}
    //cout << "goods_to_id" << endl;
    //for (auto iter = goods_to_id.begin(); iter != goods_to_id.end(); ++iter)
    //{
    //    cout << iter->second << "    " << iter->first << endl;
    //}
    //cout << "cpp_goods_weight" << endl;
    //for (int i=0; i< cpp_goods_weight.size(); ++i)
    //{
    //    cout << i << "    " << cpp_goods_weight[i] << "    " << id_to_goods[i] << endl;
    //}

    //cpp_weight_measure
    vector<float> cpp_weight_measure;
    bp::list py_weight_measure = bp::extract<bp::list>(weight_measure);
    for (int i = 0; i < bp::len(py_weight_measure); ++i)
    {
        cpp_weight_measure.push_back(bp::extract<float>(py_weight_measure[i]));
    }
    //for (int i=0; i< cpp_weight_measure.size(); ++i)
    //{
    //    cout << i << "    " << cpp_weight_measure[i] << endl;
    //}

    //cpp_order
    ORDER_INFO cpp_order;
    cpp_order.modify_time = 0;
    cpp_order.is_friendly = true;
    cpp_order.weight_predict = vector<float>(1, cpp_weight_measure[0]);
    cpp_order.order = vector<int>(cpp_goods_weight.size(), 0);
    bp::dict py_order = bp::extract<bp::dict>(order);
    bp::list py_order_items = py_order.items();
    for (int i = 0; i < bp::len(py_order_items); ++i)
    {
        bp::tuple one_item = bp::extract<bp::tuple>(py_order_items[i]);
        string cpp_goods = bp::extract<string>(one_item[0]);
        int cpp_num = bp::extract<int>(one_item[1]);
        int cpp_id = goods_to_id[cpp_goods];
        cpp_order.order[cpp_id] = cpp_num;
    }
    //cout << "cpp_order" << endl;
    //cout << "modify_time = " << cpp_order.modify_time << endl;
    //cout << "is_friendly = " << cpp_order.is_friendly << endl;
    //cout << "weight_predict = ";
    //for (int i=0; i< cpp_order.weight_predict.size(); ++i)
    //{
    //    cout << cpp_order.weight_predict[i] << "  ";
    //}
    //cout << endl;
    //cout << "order = " << endl;
    //for (int i=0; i< cpp_order.order.size(); ++i)
    //{
    //    cout << i << "    " << cpp_order.order[i] << "    " << id_to_goods[i] << endl;
    //}
    //cout << "modification = " << endl;
    //for (int i=0; i< cpp_order.modification.size(); ++i)
    //{
    //    MOD_INFO &one = cpp_order.modification[i];
    //    cout << "in_out_track = " << one.in_out_track << "  ";
    //    cout << "key = " << one.key << "  ";
    //    cout << "hid = " << one.hid << "  ";
    //    cout << "cls = " << one.cls << "  ";
    //    cout << "cls_conf = " << one.cls_conf << "  ";
    //    cout << "mod_state = " << one.mod_state << "  ";
    //    cout << endl;
    //}

    //cpp_in_out_list
    vector<IN_OUT_INFO> cpp_in_out_list;
    bp::list py_in_out_list = bp::extract<bp::list>(in_out_list);
    for (int i = 0; i < bp::len(py_in_out_list); ++i)
    {
        bp::dict py_one_in_out = bp::extract<bp::dict>(py_in_out_list[i]);
        IN_OUT_INFO cpp_one_in_out;
        string cpp_in_out_track = bp::extract<string>(py_one_in_out["in_out_track"]);
        if (cpp_in_out_track == "IN")
        {
            cpp_one_in_out.in_out_track = IN_STATE;
        }
        else
        {
            cpp_one_in_out.in_out_track = OUT_STATE;
        }
        cpp_one_in_out.key = bp::extract<int>(py_one_in_out["time_stamp"]);
        cpp_one_in_out.hid = bp::extract<int>(py_one_in_out["hid"]);
        bp::list py_cls_90 = bp::extract<bp::list>(py_one_in_out["cls_90"]);
        for (int j = 0; j < bp::len(py_cls_90); ++j)
        {
            string goods = bp::extract<string>(py_cls_90[j]);
            int cpp_id = goods_to_id[goods];
            cpp_one_in_out.cls_90.push_back(cpp_id);
        }
        bp::list py_cls_conf_90 = bp::extract<bp::list>(py_one_in_out["cls_conf_90"]);
        for (int j = 0; j < bp::len(py_cls_conf_90); ++j)
        {
            float cls_conf = bp::extract<float>(py_cls_conf_90[j]);
            cpp_one_in_out.cls_conf_90.push_back(cls_conf);
        }
        cpp_in_out_list.push_back(cpp_one_in_out);
    }
    //cout << "cpp_in_out_list " << endl;
    //for (int i=0; i< cpp_in_out_list.size(); ++i)
    //{
    //    IN_OUT_INFO &one = cpp_in_out_list[i];

    //    cout << "in_out_track = " << one.in_out_track << "  ";
    //    cout << "key = " << one.key << "  ";
    //    cout << "hid = " << one.hid << "  ";
    //    cout << "cls_90 = [";
    //    for (int j = 0; j < one.cls_90.size(); ++j)
    //    {

    //        cout << id_to_goods[one.cls_90[j]] << " ";
    //    }
    //     cout<< "]  ";
    //     cout << "cls_conf_90 = [";
    //     for (int j = 0; j < one.cls_conf_90.size(); ++j)
    //     {
    //         cout << one.cls_conf_90[j] << " ";
    //     }
    //     cout << "]  ";
    //     cout << endl;
    //}

    //cpp_weight_thresh, cpp_modify_times_thresh, cpp_max_in_out_time
    float cpp_weight_thresh = bp::extract<float>(weight_thresh);
    int cpp_modify_times_thresh = bp::extract<int>(modify_times_thresh);
    int cpp_max_in_out_time = bp::extract<int>(max_in_out_time);
    //cout << "cpp_weight_thresh = " << cpp_weight_thresh << endl;
    //cout << "cpp_modify_times_thresh = " << cpp_modify_times_thresh << endl;
    //cout << "cpp_max_in_out_time = " << cpp_max_in_out_time << endl;

    //cpp_opt_order
    ORDER_INFO cpp_opt_order;
    cpp_modify_order(cpp_order, cpp_in_out_list, cpp_weight_measure, cpp_goods_weight,
        cpp_weight_thresh, cpp_modify_times_thresh, cpp_max_in_out_time, cpp_opt_order);
    //cout << "cpp_opt_order" << endl;
    //cout << "modify_time = " << cpp_opt_order.modify_time << endl;
    //cout << "is_friendly = " << cpp_opt_order.is_friendly << endl;
    //cout << "weight_predict = ";
    //for (int i = 0; i< cpp_opt_order.weight_predict.size(); ++i)
    //{
    //    cout << cpp_opt_order.weight_predict[i] << "  ";
    //}
    //cout << endl;
    //cout << "order = " << endl;
    //for (int i = 0; i< cpp_opt_order.order.size(); ++i)
    //{
    //    cout << i << "    " << cpp_opt_order.order[i] << "    " << id_to_goods[i] << endl;
    //}
    //cout << "modification = " << endl;
    //for (int i = 0; i< cpp_opt_order.modification.size(); ++i)
    //{
    //    MOD_INFO &one = cpp_opt_order.modification[i];
    //    cout << "in_out_track = " << one.in_out_track << "  ";
    //    cout << "key = " << one.key << "  ";
    //    cout << "hid = " << one.hid << "  ";
    //    cout << "cls = " << one.cls << "  ";
    //    cout << "cls_conf = " << one.cls_conf << "  ";
    //    cout << "mod_state = " << one.mod_state << "  ";
    //    cout << endl;
    //}

    //r_opt_order
    bp::dict r_opt_order;
    for (int i = HAND_ID + 1; i < cpp_opt_order.order.size(); ++i)
    {
        int cpp_num = cpp_opt_order.order[i];
        if (cpp_num > 0)
        {
            string cpp_goods = id_to_goods[i];
            r_opt_order[cpp_goods] = cpp_num;
        }
    }

    //r_opt_modification
    bp::list r_opt_modification;
    for (int i = 0; i < cpp_opt_order.modification.size(); ++i)
    {
        MOD_INFO cpp_mod = cpp_opt_order.modification[i];
        string mod_in_out_track;
        if (cpp_mod.in_out_track == IN_STATE)
        {
            mod_in_out_track = "IN";
        }
        else
        {
            mod_in_out_track = "OUT";
        }
        int mod_key = cpp_mod.key;
        int mod_hid = cpp_mod.hid;
        string mod_cls = id_to_goods[cpp_mod.cls];
        float mod_cls_conf = cpp_mod.cls_conf;
        string mod_state;
        if(cpp_mod.mod_state == SELECT)
        {
            mod_state = "SELECT";
        }
        else if(cpp_mod.mod_state == CHANGE)
        {
            mod_state = "CHANGE";
        }
        else
        {
            mod_state = "MISS";
        }
        bp::tuple py_mod = bp::make_tuple(
            mod_in_out_track,
            mod_key,
            mod_hid,
            mod_cls,
            mod_cls_conf,
            mod_state);
        r_opt_modification.append(py_mod);
    }

    bool r_opt_is_friendly = cpp_opt_order.is_friendly;

    bp::tuple r = bp::make_tuple(r_opt_order, r_opt_modification, r_opt_is_friendly);
    return r;
}



BOOST_PYTHON_MODULE(modify_order_by_weight)
{
    using namespace boost::python;
    def("modify_order", modify_order);
}

