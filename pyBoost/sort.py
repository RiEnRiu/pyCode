#-*-coding:utf-8-*-
"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import sys
import voc as pbvoc
import numpy as np
import sklearn
import filterpy 

from numba import jit
#from sklearn.utils.linear_assignment_ import linear_assignment
#from filterpy.kalman import KalmanFilter


_jit_exIouMatrix = jit(pbvoc.exIouMatrix)



class KalmanBoxTracker():
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    maxsize = sys.maxsize-1000
    def __init__(self,bbox,id=None):
        """
        Initialises a tracker using initial bounding box.
        """
        #private

        #define constant velocity model
        self.kf = filterpy.kalman.KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.pre_z = self._convert_bbox_to_z(bbox)
        self.kf.x[:4] = self.pre_z

        #public
        if id is None:
            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1
            if KalmanBoxTracker.count>KalmanBoxTracker.maxsize:
                KalmanBoxTracker.count = 0
        else:
            self.id = id
        self.age = 0
        self.predict_time_since_update = 0
        self.update_time_since_predict = 0

        self.time_since_update = 0
        self.hit_streak = 0


    def _convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
          [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
          the aspect ratio
        """
        w = bbox[2] - bbox[0] + 1 
        h = bbox[3] - bbox[1] + 1
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h    #scale is just area
        r = w / h
        return np.array([x,y,s,r]).reshape((4,1))

    def _convert_x_to_bbox(x):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
          [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w / 2.,x[1] - h / 2.,x[0] + w / 2 - 1, x[1] + h / 2 - 1]).reshape((1,4))

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.predict_time_since_update = 0
        self.update_time_since_predict += 1        
        #self.time_since_update = 0
        #self.hit_streak += 1
        self.pre_z = self._convert_bbox_to_z(bbox)
        self.kf.update(self.pre_z)
        self.age += 1
        return

    def stay(self):
        self.kf.update(self.pre_z)
        return

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.update_time_since_predict = 0
        self.predict_time_since_update += 1
        #if self.time_since_update > 0:
        #    self.hit_streak = 0
        #self.time_since_update += 1
        return self._convert_x_to_bbox(self.kf.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self._convert_x_to_bbox(self.kf.x)


def _associate_detections_to_trackers(detections,trackers,iou_threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns a list of matches_pairs
    """

    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((0,2),dtype=np.int32)

    #iou_matrix = pbvoc.exIouMatrix(detections,trackers)
    iou_matrix = _jit_exIouMatrix(detections,trackers)

    #match
    matched_pairs = sklearn.utils.linear_assignment_.linear_assignment(-iou_matrix)
    
    #del low IOU
    matches = []
    for m in matched_pairs:
        if iou_matrix[m[0],m[1]] < iou_threshold:
            matches.append(False)
        else:
            matches.append(True)
    return matched_pairs[matches].reshape(-1,2)


class SortUnlimited:
    def __init__(self,max_loss_times=5, min_age=3,iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self._max_loss_times = max_loss_times
        self._min_age = min_age
        self._trackers = {}
        self._iou_threshold = iou_threshold


    def update(self,dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns (id list for dets, matched trk_predict dict, unmatched trk_predict dict), id = -1 means unmatched
        """
        #get predicted locations from existing _trackers and _reserve_trackers
        trk_predict = [] #[xmin, ymin, xmax, ymax, trk_id]
        for id in set(self._trackers):
            one_trk = self._trackers[id]
            pos = one_trk.predict()[0]
            if np.any(np.isnan(pos)):
                self._trackers.pop(id)
            else:
                _l = [pos[0], pos[1], pos[2], pos[3], id]
                trk_predict.append(_l)

        #np.array for match
        if len(dets) == 0:
            np_dets = np.empty((0,4),np.int32)
        else:
            np_dets = np.array(dets,np.int32)[:,0:4]

        if len(trk_predict) == 0:
            np_trk_predict = np.empty((0,4),np.int32)
        else:
            np_trk_predict = np.array(trk_predict,np.int32)[:,0:4]

        #match
        matched_pairs = _associate_detections_to_trackers(np_dets, np_trk_predict, self._iou_threshold)

        #update matched tracker
        for p in matched_pairs:
            # p = [i_dets, i_trk_predict]
            trk_id = trk_predict[p[1]][-1]
            self._trackers[trk_id].update(dets[p[0]])

        #id output
        out_id_list = [-1]*len(dets)
        for p in matched_pairs:
            # p = [i_dets, i_trk_predict]
            out_id_list[p[0]] = trk_predict[p[1]][-1]

        #create and initialise new trackers for unmatched detections
        i_unmatched_dets = set(range(len(dets)))-matched_pairs[:,0]
        for i in i_unmatched_dets:
            trk = KalmanBoxTracker(np_dets[i,0:4])
            self._reserve_trackers[trk.id] = trk

        #move self._reserve_trackers[] whose .hit_streak >= self._min_hits to _trackers
        for id in set(self._reserve_trackers):
            one_trk = self._reserve_trackers[id]
            if one_trk.hit_streak >= self._min_hits:
                self._trackers[id] = self._reserve_trackers.pop(id)



        #del trackers over age
        for id in set(self._trackers):
            one_trk = self._trackers[id]
            if one_trk.time_since_update>self._max_age:
                self._trackers.pop(id)
        for id in set(self._reserve_trackers):
            one_trk = self._reserve_trackers[id]
            if one_trk.time_since_update>self._max_age:
                self._reserve_trackers.pop(id)
        
        #output
        matched_trk_predict = {id:trk_predict_dict[id] for id in out_id_list if id!=-1}
        ummatched_trk_predict = {id:trk_predict_dict[id] for id in trk_predict_dict.keys() - matched_trk.keys()}

        return out_id_list, matched_trk_predict, ummatched_trk_predict
    
    def stay_unmatched(self):
        for one_trk in self._trackers.values:
            if one_trk.time_since_update > 0:
                one_trk.stay()
        for one_trk in self._reserve_trackers.values:
            if one_trk.time_since_update > 0:
                one_trk.stay()
        return



        
