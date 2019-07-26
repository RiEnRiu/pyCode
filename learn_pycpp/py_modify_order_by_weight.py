def _lcs_native(list1,list2,threshold):
    len1,len2 = len(list1),len(list2)
    matrix = [[0 for i in range(len2 + 1)] for j in range(len1 + 1)]
    for i in range(len1):
        for j in range(len2):
            if abs(list1[i] - list2[j]) < threshold:
                matrix[i + 1][j + 1] = matrix[i][j] + 1
            else:
                matrix[i + 1][j + 1] = max(matrix[i][j + 1], matrix[i + 1][j])
    return matrix[len(list1)][len(list2)]

def _weight_match_times(weight_predict, weight_measure, weight_thresh):
    return _lcs_native(weight_predict, weight_measure, weight_thresh)

def _find_max_wmt(index_range, weight_predicts, weight_measure, weight_thresh):
    r_index = []
    max_wmt = -1
    for i in index_range:
        weight_predict = weight_predicts[i]
        wmt = _weight_match_times(weight_predict, weight_measure, weight_thresh)
        if max_wmt < wmt:
            r_index = [i]
            max_wmt = wmt
        elif max_wmt == wmt:
            r_index.append(i)
    return r_index

def _find_min_mod_multicls(index_range, modify_timess, modifications):
    r_index = []
    min_modify_times = 10000000
    min_modification = 10000000
    for i in index_range:
        modify_times = modify_timess[i]
        modification = len(modifications[i])
        if min_modify_times > modify_times:
            r_index = [i]
            min_modify_times = modify_times
            min_modification = modification
        elif min_modify_times == modify_times:
            if min_modification > modification:
                r_index = [i]
                min_modify_times = modify_times
                min_modification = modification
            elif min_modification == modification:
                r_index.append(i)
    return r_index

def _split_friendly(index_range, is_friendlys):
    index_friendly = []
    index_not_friendly = []
    for i in index_range:
        if is_friendlys[i]:
            index_friendly.append(i)
        else:
            index_not_friendly.append(i)
    return index_friendly, index_not_friendly


def _find_min_dw(index_range, weight_predicts, weight_measure):
    r_index = index_range[0]
    min_dw = abs(weight_predicts[r_index][-1] - weight_measure[-1])
    for i in index_range[1:]:
        abs_dw = abs(weight_predicts[i][-1] - weight_measure[-1])
        if abs_dw < min_dw:
            min_dw = abs_dw
            r_index = i
    return r_index


def _choice_optimum_multicls(orders, modifications, modify_timess, \
                             weight_predicts, is_friendlys, \
                             weight_measure, weight_thresh):
    # 1st. find low weight thresh
    index_low_weight_thresh = []
    index_over_weight_thresh = []
    ps = ""
    for i, weight_predict in enumerate(weight_predicts):
        if abs(weight_predict[-1] - weight_measure[-1]) < weight_thresh:
            ps = ps+"{0} * ".format(int(abs(weight_predict[-1] - weight_measure[-1])*1000))
            index_low_weight_thresh.append(i)
        else:
            index_over_weight_thresh.append(i)
    #print(ps)
    #print(weight_thresh)
    #print(weight_predicts[-1])
    #print(weight_measure)
    #print(len(index_low_weight_thresh),len(index_over_weight_thresh))

    # 2nd. return visual case if all over weight_thresh
    if len(index_low_weight_thresh) == 0:
        for i in range(len(orders)):
            if len(modifications[i]) == 0:
                return orders[i], modifications[i], False

    # 3rd. find max weight match times
    index_max_wmt = _find_max_wmt(index_low_weight_thresh, weight_predicts, weight_measure, weight_thresh)
    #print("index_max_wmt = "+str(len(index_max_wmt)))

    # 4th. split friendly
    index_friendly_in_max_wmt, index_not_friendly_in_max_wmt = _split_friendly(index_max_wmt, is_friendlys)

    # 5th. min modifications
    if len(index_friendly_in_max_wmt) == 0:
        index_min_mod_in_not_friendly_in_max_wmt = _find_min_mod_multicls(index_not_friendly_in_max_wmt, modify_timess, modifications)
        i = _find_min_dw(index_min_mod_in_not_friendly_in_max_wmt, weight_predicts, weight_measure)
        return orders[i], modifications[i], is_friendlys[i]
    else:
        index_min_mod_in_friendly_in_max_wmt = _find_min_mod_multicls(index_friendly_in_max_wmt, modify_timess, modifications)
        i = _find_min_dw(index_min_mod_in_friendly_in_max_wmt, weight_predicts, weight_measure)
        return orders[i], modifications[i], is_friendlys[i]


def _modify_order_one_in_out_multicls(order, modification, modify_times, weight_predict, is_friendly, \
                             one_in_out, max_modify_times, goods_weight, in_out_time, max_in_out_time,\
                             r_order, r_modification, r_modify_times, r_weight_predict, r_is_friendly):

    # one_in_out.keys = {'bb', 'bb_conf', 'det_id', 'cls_id', 'cls_conf', 'cls', \
    #                    'cls_id_90', 'cls_conf_90', 'cls_90', 'hid', \
    #                    'outside', 'cacheside', 'inside', 'in_out_flag', \
    #                    'in_out_track', 'time_stamp'}
    in_out_track = one_in_out['in_out_track']
    key = one_in_out['time_stamp']
    hid = one_in_out['hid']
    cls_90 = one_in_out['cls_90']
    cls_conf_90 = one_in_out['cls_conf_90']

    # modify += 0
    maybe_cls = set()
    all_s_order = []
    all_s_modification = []
    all_s_modify_times = []
    all_s_weight_predict = []
    all_s_is_f = []

    for i, s_cls, s_cls_conf in zip(range(len(cls_90)), cls_90, cls_conf_90):
        maybe_cls.add(s_cls)
        tmp_order = order.copy()
        tmp_modification = modification.copy()
        tmp_weight_predict = weight_predict.copy()
        tmp_is_f = is_friendly
        if i!=0:
            tmp_modification.append((in_out_track, key, hid, s_cls, s_cls_conf, 'SELECT'))
        wp = tmp_weight_predict[-1]
        if s_cls == 'hand':
            have_hand = True
        else:
            if in_out_track == 'IN':
                tmp_order[s_cls] -= 1
                wp += goods_weight[s_cls]
            else:  # elif in_out_track=='OUT':
                tmp_order[s_cls] += 1
                wp -= goods_weight[s_cls]
            if tmp_order[s_cls] < -0.1:
                tmp_is_f = False
        tmp_weight_predict.append(wp)
        all_s_order.append(tmp_order)
        all_s_modification.append(tmp_modification)
        all_s_modify_times.append(modify_times)
        all_s_weight_predict.append(tmp_weight_predict)
        all_s_is_f.append(tmp_is_f)
    if len(modification) >= max_modify_times or in_out_time >= max_in_out_time:
        r_order.extend(all_s_order)
        r_modification.extend(all_s_modification)
        r_modify_times.extend(all_s_modify_times)
        r_weight_predict.extend(all_s_weight_predict)
        r_is_friendly.extend(all_s_is_f)
        return

    modify_times += 1
    # change to hand
    if 'hand' not in maybe_cls:
        tmp_order = order.copy()
        tmp_modification = modification.copy()
        tmp_weight_predict = weight_predict.copy()
        tmp_is_f = is_friendly
        tmp_modification.append((in_out_track, key, hid, 'hand', 0.111111, 'CHANGE'))
        tmp_weight_predict.append(tmp_weight_predict[-1])
        all_s_order.append(tmp_order)
        all_s_modification.append(tmp_modification)
        all_s_modify_times.append(modify_times)
        all_s_weight_predict.append(tmp_weight_predict)
        all_s_is_f.append(tmp_is_f)
    # change to other goods
    for goods in goods_weight.keys():
        if goods in maybe_cls:
            continue
        tmp_order = order.copy()
        tmp_modification = modification.copy()
        tmp_weight_predict = weight_predict.copy()
        tmp_is_f = is_friendly
        tmp_modification.append((in_out_track, key, hid, goods, 0.111111, 'CHANGE'))
        wp = tmp_weight_predict[-1]
        if in_out_track == 'IN':
            tmp_order[goods] -= 1
            wp += goods_weight[goods]
        else:  # elif in_out_track=='OUT':
            tmp_order[goods] += 1
            wp -= goods_weight[goods]
        if tmp_order[goods] < -0.1:
            tmp_is_f = False
        tmp_weight_predict.append(wp)
        all_s_order.append(tmp_order)
        all_s_modification.append(tmp_modification)
        all_s_modify_times.append(modify_times)
        all_s_weight_predict.append(tmp_weight_predict)
        all_s_is_f.append(tmp_is_f)
    r_order.extend(all_s_order)
    r_modification.extend(all_s_modification)
    r_modify_times.extend(all_s_modify_times)
    r_weight_predict.extend(all_s_weight_predict)
    r_is_friendly.extend(all_s_is_f)
    if len(modification)+1 >= max_modify_times:
        return

    # miss one goods
    for _o,_m,_mt,_wp,_if in zip(all_s_order, all_s_modification,\
                                 all_s_modify_times, all_s_weight_predict, all_s_is_f):
        for goods in goods_weight.keys():
            tmp_order = _o.copy()
            tmp_modification = _m.copy()
            tmp_weight_predict = _wp.copy()
            tmp_is_f = _if
            tmp_modification.append((in_out_track, key, hid, goods, 0.111111, 'MISS'))
            wp = tmp_weight_predict[-1]
            if in_out_track == 'IN':
                tmp_order[goods] -= 1
                wp += goods_weight[goods]
            else:  # elif in_out_track=='OUT':
                tmp_order[goods] += 1
                wp -= goods_weight[goods]
            if tmp_order[goods] < -0.1:
                tmp_is_f = False
            tmp_weight_predict[-1] = wp
            r_order.append(tmp_order)
            r_modification.append(tmp_modification)
            r_modify_times.append(_mt+1)
            r_weight_predict.append(tmp_weight_predict)
            r_is_friendly.append(tmp_is_f)
    return


def _get_no_change_add_weight(in_out_list, goods_weight):
    # one_in_out.keys = {'bb', 'bb_conf', 'det_id', 'cls_id', 'cls_conf', 'cls', \
    #                    'cls_id_90', 'cls_conf_90', 'cls_90', 'hid', \
    #                    'outside', 'cacheside', 'inside', 'in_out_flag', \
    #                    'in_out_track', 'time_stamp'}

    reversed_result = []
    this_add_weight = 0
    for one_in_out in reversed(in_out_list):
        in_out_track = one_in_out['in_out_track']
        cls = one_in_out['cls']
        if cls == 'hand':
            reversed_result.append(this_add_weight)
            continue
        if in_out_track == 'IN':
            this_add_weight += goods_weight[cls]
        else:  # elif in_out_track=='OUT':
            this_add_weight -= goods_weight[cls]
        reversed_result.append(this_add_weight)
    return reversed(reversed_result)

def modify_order_multicls(order, in_out_list,\
                         weight_measure, goods_weight, \
                         weight_thresh = 20, max_modify_times = 3,
                         max_in_out_time=30):

    '''
    # Parameters below are about order made in ther previous stage.
    order = {'yy':2,'mlqc':1, 'nfsq':3, ...}  # the number of goods taken away
    # Goods in-out in this stage
    # one_in_out.keys = {'bb', 'bb_conf', 'det_id', 'cls_id', 'cls_conf', 'cls', \
    #                    'cls_id_90', 'cls_conf_90', 'cls_90', 'hid', \
    #                    'outside', 'cacheside', 'inside', 'in_out_flag', \
    #                    'in_out_track', 'time_stamp'}

    # The weight measured by weight
    weight_measure = [4777.0, 4300.1, 3300.2, ...]

    # The real weight of goods
    goods_weight = {'yy':285.0, 'mlqc':532.0}
    '''

    tmp_order = {goods:0 for goods in goods_weight.keys()}
    tmp_order.update(order)
    orders = [tmp_order]
    modifications = [[]]
    modify_timess = [0]
    is_friendlys = [True]
    weight_predicts = [[weight_measure[0]]]

    no_modification_add_weights = _get_no_change_add_weight(in_out_list, goods_weight)
    # print(list(no_modification_add_weights))
    # make all maybe case with modify-times limit
    # t1 = time.time()
    in_out_time = 0
    for one_in_out,add_weight in zip(in_out_list, no_modification_add_weights):
        tmp_orders = []
        tmp_modifications = []
        tmp_modify_timess = []
        tmp_weight_predicts = []
        tmp_is_friendlys = []
        for order, modification, modify_times, weight_predict, is_friendly in \
                zip(orders, modifications, modify_timess, weight_predicts, is_friendlys):
            _modify_order_one_in_out_multicls(order, modification, modify_times, \
                                             weight_predict, is_friendly, \
                                             one_in_out, max_modify_times, goods_weight, \
                                             in_out_time, max_in_out_time, \
                                             tmp_orders, tmp_modifications, tmp_modify_timess, \
                                             tmp_weight_predicts, tmp_is_friendlys)
        orders.clear()
        modifications.clear()
        modify_timess.clear()
        weight_predicts.clear()
        is_friendlys.clear()
        orders = tmp_orders
        modifications = tmp_modifications
        modify_timess = tmp_modify_timess
        weight_predicts = tmp_weight_predicts
        is_friendlys = tmp_is_friendlys
        in_out_time += 1
        #print(in_out_time, len(tmp_orders))
        # input()

    # choice optimum case
    opt_order, opt_modification, opt_is_friendly = \
        _choice_optimum_multicls(orders, modifications, modify_timess, \
                                 weight_predicts, is_friendlys, \
                                 weight_measure, weight_thresh)

    # remove zero and negative in opt_order
    for goods in set(opt_order.keys()):
        if opt_order[goods] < 0.1:
            opt_order.pop(goods)

    return opt_order, opt_modification, opt_is_friendly

