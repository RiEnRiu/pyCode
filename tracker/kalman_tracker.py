import cv2
import numpy as np

def mousemove(event, x, y, s, p):
    #update 
    p['last_prediction'] = p['current_prediction']
    p['last_measurement'] = p['current_measurement']
    p['current_measurement'] = np.array([[np.float32(x)], [np.float32(y)]])
    p['kalman'].correct(p['current_measurement'])
    p['current_prediction'] = p['kalman'].predict()

    #points
    lmx, lmy = p['last_measurement'][0], p['last_measurement'][1]
    cmx, cmy = p['current_measurement'][0], p['current_measurement'][1]
    lpx, lpy = p['last_prediction'][0], p['last_prediction'][1]
    cpx, cpy = p['current_prediction'][0], p['current_prediction'][1]

    #print
    cv2.line(p['frame'], (lmx, lmy), (cmx, cmy), (255, 255, 255))
    cv2.line(p['frame'], (lpx, lpy), (cpx, cpy), (0, 0, 255))

if __name__=='__main__':
    #kalman
    kalman = cv2.KalmanFilter(4, 2) # 4：number of states，(x，y，dx，dy); 2：observed value，(x,y)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) #System measurement matrix
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) #State transition matrix
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 #Covariance of system process noise

    #show
    frame = np.zeros((700, 700, 3), np.uint8) 

    last_measurement = np.array((2, 1), np.float32)
    current_measurement = np.array((2, 1), np.float32)
    last_prediction = np.zeros((2, 1), np.float32)
    current_prediction = np.zeros((2, 1), np.float32)

    cv2.namedWindow("kalman_tracker")
    param_MouseCallback = {'kalman':kalman,'frame':frame,'current_measurement':current_measurement,\
        'last_measurement':last_measurement,'current_prediction':current_prediction,'last_prediction':last_prediction}
    cv2.setMouseCallback("kalman_tracker", mousemove,param_MouseCallback)

    while True:
        cv2.imshow("kalman_tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key==27:
            break
    cv2.destroyAllWindows()
