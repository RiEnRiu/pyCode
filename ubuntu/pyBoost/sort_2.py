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

from numba import jit
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter

#from vocBoost import calEXIOU
#_jit_exiou = jit(calEXIOU)
@jit
def _jit_exiou(box1,box2):
    #inters
    ixmin = max(box1[0], box2[0])
    iymin = max(box1[1], box2[1])
    ixmax = min(box1[2], box2[2])
    iymax = min(box1[3], box2[3])
    iw = ixmax - ixmin + 1
    ih = iymax - iymin + 1
    flag = 1
    if(iw < 0):
        iw = -iw
        flag = -1
    if(ih < 0):
        ih = -ih
        flag = -1
    inters = iw * ih
    # union
    uni = ((box2[2] - box2[0] + 1.) * (box2[3] - box2[1] + 1.) + (box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) - flag * (inters))
    return flag * inters / uni


@jit
def _jit_exiou_limited_area(box1,box2):
    exiouvalue = _jit_exiou(box1,box2)
    area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    area2 = (box2[2] - box2[0])*(box2[3] - box2[1]) 
    areavalue = min(area1,area2)/max(area1,area2)*2-1
    return min(exiouvalue,areavalue)



def _is_in_alert(bb,mask,thresh = 10):
    return mask[bb[1]:bb[3]+1,bb[0]:bb[2]+1].sum()>thresh




def _convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x,y,s,r]).reshape((4,1))

def _convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
        return np.array([x[0] - w / 2.,x[1] - h / 2.,x[0] + w / 2.,x[1] + h / 2.]).reshape((1,4))
    else:
        return np.array([x[0] - w / 2.,x[1] - h / 2.,x[0] + w / 2.,x[1] + h / 2.,score]).reshape((1,5))

class _KalmanBoxTracker:
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox,id=None):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self._pre_z = _convert_bbox_to_z(bbox)
        self.kf.x[:4] = self._pre_z

        self.time_since_update = 0
        if(id is None):
            self.id = _KalmanBoxTracker.count
            _KalmanBoxTracker.count += 1
        else:
            self.id = id
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0



    
    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(_convert_bbox_to_z(bbox))
    
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(_convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def stay(self):
        self.kf.update(self._pre_z)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return _convert_x_to_bbox(self.kf.x)


def _associate_detections_to_trackers(detections,trackers,iou_threshold):#iou_threshold = 0.3
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns a list of matches_pairs
    """
    if(len(trackers) == 0 or len(detections) == 0):
        return np.empty((0,2),dtype=int)

    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    
    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = _jit_exiou(det,trk)
            # iou_matrix[d,t] = _jit_exiou_limited_area(det,trk)
    #match
    matched_pairs = linear_assignment(-iou_matrix)
    
    #del low IOU
    matches = []
    for m in matched_pairs:
        if(iou_matrix[m[0],m[1]] < iou_threshold):
            matches.append(False)
        else:
            matches.append(True)
    output = matched_pairs[matches]
    if(len(output) == 0):
        return np.empty((0,2),dtype=int)
    else:
        return output

class SortUnlimited:
    def __init__(self,max_age=1,min_hits=3,iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self._max_age = max_age
        self._min_hits = min_hits
        
        self._trackers = []
        self._iou_threshold = iou_threshold
        
    def update(self,dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns (dets with id in last column, trk prediction with id in last column)， id = -1 means unmatched
        """
        #get predicted locations from existing _trackers.
        trks = []
        to_del = []
        for t,one_trk in enumerate(self._trackers):
            pos = one_trk.predict()[0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
            else:
                trks.append(np.array([pos[0], pos[1], pos[2], pos[3], one_trk.id]))
        for t in reversed(to_del):
            self._trackers.pop(t)

        if(len(dets) == 0):
            out_dets = np.empty((0,5)).astype(np.int)
        else:
            out_dets = (-np.ones((len(dets),5))).astype(np.int)
            out_dets[:,0:4] = dets[:,0:4]  

        if(len(trks) == 0):
            trks = np.empty((0,5)).astype(np.int)
        else:
            trks = np.array(trks,np.int)

       #trks = [[b[0],b[1],b[2]，b[3], trackerID],...]
        #print(out_dets)
        #print(trks)

        #match
        matched_pairs = _associate_detections_to_trackers(out_dets[:,0:4],trks[:,0:4],self._iou_threshold)

        #updata matched tracker
        for p in matched_pairs:
            # p = [i_dets, i_trackers]
            self._trackers[p[1]].update(out_dets[p[0]])

        #get id for out_dets
        for p in matched_pairs:
            # p = [i_out_dets, i_trks]
            out_dets[p[0],-1] = trks[p[1],-1]

        #create and initialise new trackers for unmatched detections
        unmatched_i_dets = np.where(out_dets[:,-1] == -1)[0]
        for i in unmatched_i_dets:
            trk = _KalmanBoxTracker(out_dets[i,:]) 
            self._trackers.append(trk)

        ############################################################################
        # #del self._trackers[].hit_streak < self._min_hits
        # for p in matched_pairs:
        #     # p = [i_out_dets, i_trackers]
        #     if(self._trackers[p[1]].hit_streak < self._min_hits):
        #         out_dets[p[0],-1] = -1
        ############################################################################


        #del trackers over age
        trk_to_del = [i for i,trk in enumerate(self._trackers) \
                    if trk.time_since_update > self._max_age]
        for i in reversed(trk_to_del):
            self._trackers.pop(i)

        return out_dets, {int(x[-1]):x[0:4].astype(np.int) for x in trks}

    def update_only_id(self,dets):
        out_dets,out_trks = self.update(dets)
        return [x[-1] for x in out_dets], out_trks

    def stay_unmatched(self):
        for trk in self._trackers:
            if(trk.time_since_update > 0):
                trk.stay()
        return



class SortLimited:
    def __init__(self,max_age=1,min_hits=3,iou_threshold=0.3,max_num_trks=2):
        """
        Sets key parameters for SORT
        """
        self._max_age = max_age
        self._min_hits = min_hits
        self._working_trks = []
        self._trackers = [None] * max_num_trks
        self._iou_threshold = iou_threshold
        
    def update(self,dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns (dets with id in last column, trk prediction with id in last column)， id = -1 means unmatched
        """
        #get predicted locations from existing __trackers.
        trks = []
        to_del = []
        for t in self._working_trks:
            one_trk = self._trackers[t]
            pos = one_trk.predict()[0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
            else:
                trks.append(np.array([pos[0], pos[1], pos[2], pos[3], one_trk.id]))
        for t in reversed(to_del):
            self._trackers[t] = None
            self._working_trks.remove(t)

        #print('trks = ' + str(trks)+',    len(_working_trks) = '+str(len(self._working_trks)))

        if(len(dets) == 0):
            out_dets = np.empty((0,5)).astype(np.int)
        else:
            out_dets = (-np.ones((len(dets),5))).astype(np.int)
            out_dets[:,0:4] = dets[:,0:4]  

        if(len(trks) == 0):
            trks = np.empty((0,5)).astype(np.int)
        else:
            trks = np.array(trks,np.int)

        #match
        matched_pairs = _associate_detections_to_trackers(out_dets[:,0:4],trks[:,0:4],self._iou_threshold)
        #print(out_dets,trks,matched_pairs)


        #updata matched tracker
        for p in matched_pairs:
            # p = [i_out_dets, i_working_trks]
            i_trackers = self._working_trks[p[1]]
            self._trackers[i_trackers].update(out_dets[p[0]])

        #get id for out_dets
        for p in matched_pairs:
            # p = [i_out_dets, i_trks]
            out_dets[p[0],-1] = trks[p[1],-1]

        #create and initialise new trackers for unmatched detections
        unmatched_i_dets = np.where(out_dets[:,-1] == -1)[0]
        try:
            for i in unmatched_i_dets:

                index = self._trackers.index(None)
                trk = _KalmanBoxTracker(dets[i,:],index) 
                #print('create one')
                self._trackers[index] = trk
                self._working_trks.append(index)
        except:
            pass
           
       #del self._trackers[].hit_streak < self._min_hits
        for p in matched_pairs:
            # p = [i_out_dets, i_working_trks]
            i_trackers = self._working_trks[p[1]]
            if(self._trackers[i_trackers].hit_streak < self._min_hits):
                out_dets[p[0],-1] = -1
            
        #del trackers over age
        trk_to_del = [i for i in self._working_trks \
                    if self._trackers[i].time_since_update > self._max_age]
        for i in reversed(trk_to_del):
            self._trackers[i] = None
            self._working_trks.remove(i)

        return out_dets, {int(x[-1]):x[0:4].astype(np.int) for x in trks}


    def update_only_id(self,dets):
        out_dets,out_trks = self.update(dets)

        return [x[-1] for x in out_dets], out_trks


class SortWaitOutAlert:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, alert_mask = None):
        """
        Sets key parameters for SORT
        """
        self._max_age = max_age
        self._min_hits = min_hits

        self._trackers = []
        self._iou_threshold = iou_threshold

        self._alert = alert_mask
        self._in_alert_tackers = {}

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns (dets with id in last column, trk prediction with id in last column)， id = -1 means unmatched
        """
        # get predicted locations from existing _trackers.
        trks = []
        to_del = []
        for t, one_trk in enumerate(self._trackers):
            pos = one_trk.predict()[0]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
            else:
                trks.append(np.array([pos[0], pos[1], pos[2], pos[3], one_trk.id]))
        for t in reversed(to_del):
            self._trackers.pop(t)

        if (len(dets) == 0):
            out_dets = np.empty((0, 5)).astype(np.int)
        else:
            out_dets = (-np.ones((len(dets), 5))).astype(np.int)
            out_dets[:, 0:4] = dets[:, 0:4]

        if (len(trks) == 0):
            trks = np.empty((0, 5)).astype(np.int)
        else:
            trks = np.array(trks, np.int)

        # match
        matched_pairs = _associate_detections_to_trackers(out_dets[:, 0:4], trks[:, 0:4], self._iou_threshold)

        # unmatched_tracker_index = set(range(len(self._trackers))) - set(p[1] for p in matched_pairs)
        # # update ummatched alerting tracker
        # for p1 in unmatched_tracker_index:
        #     find_in_alert_tackers = self._in_alert_tackers.get(p1)
        #     if(find_in_alert_tackers is not None):
        #         self._trackers[p1].update(find_in_alert_tackers)

        # updata matched tracker and deal alert
        for p in matched_pairs:
            # p = [i_dets, i_trackers]
            self._trackers[p[1]].update(out_dets[p[0]])
            if(_is_in_alert(out_dets[p[0]],self._alert,10)):
                self._in_alert_tackers[p[1]] = out_dets[p[0]]
            else:
                find_in_alert_tackers = self._in_alert_tackers.get(p[1])
                if(find_in_alert_tackers is not None):
                    self._in_alert_tackers.pop(p[1])

        # get id for out_dets
        for p in matched_pairs:
            # p = [i_out_dets, i_trks]
            out_dets[p[0], -1] = trks[p[1], -1]

        # create and initialise new trackers for unmatched detections
        unmatched_i_dets = np.where(out_dets[:, -1] == -1)[0]
        for i in unmatched_i_dets:
            trk = _KalmanBoxTracker(out_dets[i, :])
            self._trackers.append(trk)

        # del self._trackers[].hit_streak < self._min_hits
        for p in matched_pairs:
            # p = [i_out_dets, i_trackers]
            if (self._trackers[p[1]].hit_streak < self._min_hits):
                out_dets[p[0], -1] = -1

        # del trackers over age
        trk_to_del = [i for i, trk in enumerate(self._trackers) \
                      if trk.time_since_update > self._max_age]
        for i in reversed(trk_to_del):
            self._trackers.pop(i)

        return out_dets, {int(x[-1]): x[0:4].astype(np.int) for x in trks}
