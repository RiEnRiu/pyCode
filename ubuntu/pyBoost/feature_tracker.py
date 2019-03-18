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





import numpy as np
import sklearn
import filterpy 

def similarity(feature1,feature2):
    return 0


def _associate_new_features_to_old(new_features, old_features, threshold):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns a list of matches_pairs
    """

    if(len(new_features) == 0 or len(old_features) == 0):
        return np.empty((0,2),dtype=np.int32)

    similarity_matrix = np.zeros((len(new_features),len(old_features)),dtype=np.float64)
    for i1,f1 in enumerate(new_features):
        for i2,f2 in enumerate(old_features):
            similarity_matrix[i1,i2] = similarity(f1,f2)

    #match
    matched_pairs = sklearn.utils.linear_assignment_.linear_assignment(-similarity_matrix)
    
    #del low thresh
    matches = []
    for m in matched_pairs:
        if(similarity_matrix[m[0],m[1]] < threshold):
            matches.append(False)
        else:
            matches.append(True)
    return matched_pairs[matches].reshape(-1,2)

class featureTracker():
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,feature,id=None):
        """
        Initialises a tracker using initial bounding box.
        """
        #private
        self.pre_feature = feature

        #public
        self.time_since_update = 0
        if(id is None):
            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1
        else:
            self.id = id
        self.hit_streak = 0

    def update(self,feature):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.pre_feature = feature
        return

    def preFeature(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        return self.pre_feature

class SortFeature():
    def __init__(self,max_age=5,min_hits=3,threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self._max_age = max_age
        self._min_hits = min_hits
       
        self._trackers = []
        self._reserve_trackers = []

        self._threshold = threshold


    def update(self,features):
        """
        Params:
          features - a numpy array of detections in the format [features1,features2,...]
        Requires: this method must be called once for each frame even with empty features.
        Returns [id], id = -1 means unmatched
        """
        #get predicted locations from existing _trackers and _reserve_trackers
        trk_pre_feature = [trk.preFeature() for trk in self._trackers]
        len_trackers = len(trk_pre_feature)
        trk_pre_feature.extend([trk.preFeature() for trk in self._reserve_trackers])

        #match
        matched_pairs = _associate_new_features_to_old(features,trk_pre_feature,self._threshold)

        #update matched tracker
        for i_new, i_old in matched_pairs:
            if(i_old<len_trackers):
                self._trackers[i_old].update(features[i_new])
            else:
                self._reserve_trackers[i_old-len_trackers].update(features[i_new])
         
        #id output
        out_id_list = [-1]*len(features)
        for i_new, i_old in matched_pairs:
            if(i_old<len_trackers):
                trk = self._trackers[i_old]
            else:
                trk = elf._reserve_trackers[i_old-len_trackers]
            out_id_list[i_new] = trk.id

        #move self._reserve_trackers[] whose .hit_streak >= self._min_hits to _trackers
        to_move = []
        for i,trk in enumerate(self._reserve_trackers):
            if(trk.hit_streak >= self._min_hits):
                to_move.append(i)
        for i in reversed(to_move):
            self._trackers.append(self._reserve_trackers.pop(i))

        #create and initialise new trackers for unmatched feature
        unmatched_i_dets = set(range(len(features)))-matched_pairs[:,0]
        for i in unmatched_i_dets:
            trk = featureTracker(features[i])
            self._reserve_trackers.append(trk)

        #del trackers over age
        to_del = []
        for i,trk in enumerate(self._trackers):
            if(trk.time_since_update>self._max_age):
                to_del.append(i)
        for i in reversed(to_del):
            self._trackers.pop(i)
        to_del = []
        for i,trk in enumerate(self._reserve_trackers):
            if(trk.time_since_update>self._max_age):
                to_del.append(i)
        for i in reversed(to_del):
            self._reserve_trackers.pop(i)
        
        return out_id_list


