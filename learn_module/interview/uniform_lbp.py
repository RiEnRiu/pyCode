import numpy as np
import cv2

_dict_256_to_59 = {\
    0:0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 58, 6: 5, \
    7: 6, 8: 7, 9: 58, 10: 58, 11: 58, 12: 8, \
    13: 58, 14: 9, 15: 10, 16: 11, 17: 58, 18: 58, \
    19: 58, 20: 58, 21: 58, 22: 58, 23: 58, 24: 12, \
    25: 58, 26: 58, 27: 58, 28: 13, 29: 58, 30: 14, \
    31: 15, 32: 16, 33: 58, 34: 58, 35: 58, 36: 58, \
    37: 58, 38: 58, 39: 58, 40: 58, 41: 58, 42: 58, \
    43: 58, 44: 58, 45: 58, 46: 58, 47: 58, 48: 17, \
    49: 58, 50: 58, 51: 58, 52: 58, 53: 58, 54: 58, \
    55: 58, 56: 18, 57: 58, 58: 58, 59: 58, 60: 19, \
    61: 58, 62: 20, 63: 21, 64: 22, 65: 58, 66: 58, \
    67: 58, 68: 58, 69: 58, 70: 58, 71: 58, 72: 58, \
    73: 58, 74: 58, 75: 58, 76: 58, 77: 58, 78: 58, \
    79: 58, 80: 58, 81: 58, 82: 58, 83: 58, 84: 58, \
    85: 58, 86: 58, 87: 58, 88: 58, 89: 58, 90: 58, \
    91: 58, 92: 58, 93: 58, 94: 58, 95: 58, 96: 23, \
    97: 58, 98: 58, 99: 58, 100: 58, 101: 58, 102: 58, \
    103: 58, 104: 58, 105: 58, 106: 58, 107: 58, 108: 58, \
    109: 58, 110: 58, 111: 58, 112: 24, 113: 58, 114: 58, \
    115: 58, 116: 58, 117: 58, 118: 58, 119: 58, 120: 25, \
    121: 58, 122: 58, 123: 58, 124: 26, 125: 58, 126: 27, \
    127: 28, 128: 29, 129: 30, 130: 58, 131: 31, 132: 58, \
    133: 58, 134: 58, 135: 32, 136: 58, 137: 58, 138: 58, \
    139: 58, 140: 58, 141: 58, 142: 58, 143: 33, 144: 58, \
    145: 58, 146: 58, 147: 58, 148: 58, 149: 58, 150: 58, \
    151: 58, 152: 58, 153: 58, 154: 58, 155: 58, 156: 58, \
    157: 58, 158: 58, 159: 34, 160: 58, 161: 58, 162: 58, \
    163: 58, 164: 58, 165: 58, 166: 58, 167: 58, 168: 58, \
    169: 58, 170: 58, 171: 58, 172: 58, 173: 58, 174: 58, \
    175: 58, 176: 58, 177: 58, 178: 58, 179: 58, 180: 58, \
    181: 58, 182: 58, 183: 58, 184: 58, 185: 58, 186: 58, \
    187: 58, 188: 58, 189: 58, 190: 58, 191: 35, 192: 36, \
    193: 37, 194: 58, 195: 38, 196: 58, 197: 58, 198: 58, \
    199: 39, 200: 58, 201: 58, 202: 58, 203: 58, 204: 58, \
    205: 58, 206: 58, 207: 40, 208: 58, 209: 58, 210: 58, \
    211: 58, 212: 58, 213: 58, 214: 58, 215: 58, 216: 58, \
    217: 58, 218: 58, 219: 58, 220: 58, 221: 58, 222: 58, \
    223: 41, 224: 42, 225: 43, 226: 58, 227: 44, 228: 58, \
    229: 58, 230: 58, 231: 45, 232: 58, 233: 58, 234: 58, \
    235: 58, 236: 58, 237: 58, 238: 58, 239: 46, 240: 47, \
    241: 48, 242: 58, 243: 49, 244: 58, 245: 58, 246: 58, \
    247: 50, 248: 51, 249: 52, 250: 58, 251: 53, 252: 54, \
    253: 55, 254: 56, 255: 57}
  
def lbp256_to_lbp59(feature):
    lbp = np.zeros(59)
    for f in feature:
        for v in f:
            lbp[_dict_256_to_59[v]] += 1
    return lbp/feature.size
  
_order_kernel = np.array([[128,64,32],[1,0,16],[2,4,8]],np.uint8)
_inversed_order_kernel = np.array([[128,1,2],[64,0,4],[32,16,8]],np.uint8)

def uniformLBP(gray):
    global _order_kernel
    ker = _order_kernel
    # global _inversed_order_kernel
    # ker = _inversed_order_kernel

    h,w = gray.shape
    lbp256 = np.zeros((h,w),np.uint8)
    padding = np.zeros((h+2,w+2),gray.dtype)
    padding[1:-1,1:-1] = gray

    for i in range(h):
        for j in range(w):
            center = padding[i+1,j+1]
            neighbor = padding[i:i+3,j:j+3]
            temp = (neighbor>=center)*ker
            # temp = (neighbor>center)*ker
            lbp256[i,j] = temp.sum()

    lbp59 = lbp256_to_lbp59(lbp256)
    return lbp256, lbp59


if __name__=='__main__':
    gray = cv2.imread('./Mrma.png',cv2.IMREAD_GRAYSCALE)
    lbp256, lbp59 = uniformLBP(gray)
    cv2.imshow('lbp256',lbp256)
    print(lbp59)
    cv2.waitKey(0)

