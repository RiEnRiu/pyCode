import numpy as np

def conv2(image_c1,ker,size_type='full'):
    img = np.array(image_c1,np.float32)
    K = np.array(ker,np.float32)

    #size
    himg = img.shape[0]
    wimg = img.shape[1]
    hk = K.shape[0]
    wk = K.shape[1]

    #pading image size
    hpadimg = himg+hk*2-2
    wpadimg = wimg+wk*2-2

    #reverse ker
    rK = K.tolist()
    for line in rK:
        line.reverse()
    rK.reverse()

    #init with 0
    padimg = np.zeros((hpadimg,wpadimg),np.float32)
    full_conv2 = np.zeros((max(himg,hk,himg+hk-1),max(wimg,wk,wimg+wk-1)),np.float32)

    #calculate conv2
    padimg[(hk-1):(hk-1+himg),(wk-1):(wk-1+wimg)] = img
    for i in range(0,hpadimg-hk+1):
        for j in range(0,wpadimg-wk+1):
            full_conv2[i][j] = (padimg[i:i+hk,j:j+wk]*rK).sum()
    
    #return the user size
    if size_type=='full':
        return full_conv2.astype(image_c1.dtype)
    elif size_type=='same':
        i1 = int((full_conv2.shape[0]-img.shape[0]+1)/2)
        i2 = int((full_conv2.shape[1]-img.shape[1]+1)/2)
        return full_conv2[i1:i1+himg,i2:i2+wimg].astype(image_c1.dtype)
    elif size_type=='valid':
        i1 = hk-1
        i2 = wk-1
        return full_conv2[i1:himg,i2:wimg].astype(image_c1.dtype)
    else:
        raise AssertionError('size_type must be \"full\" or \"same\" or \"valid\"')

if __name__=='__main__':
    
    img = [\
           [   5, 126, 255, 145, 220, 106],\
           [   6,  51,  25, 254, 141, 211],\
           [ 210,  42, 227, 184, 227,  85],\
           [ 224,  15, 172, 144, 103,  30],\
           [ 198,  25, 108,  26, 184, 151],\
           [ 252, 199,  17, 134, 132,  87],\
           [  96, 198, 151,   8, 121, 121],\
           [  17, 208, 194, 232, 139, 180],\
           [  21, 215,  20,  18, 203, 247]\
          ]

    ker_3_3 = [\
               [   1,   0,   2],\
               [   4,   3,   4],\
               [   5,   1,   5]\
              ]

    ker_5_5 = [\
               [   3,   5,   5,   2,   1],\
               [   5,   2,   5,   2,   1],\
               [   3,   1,   3,   5,   2],\
               [   3,   2,   5,   1,   2],\
               [   0,   2,   2,   4,   2]\
              ]

    img = np.array(img)
    ker_3_3 = np.array(ker_3_3)
    ker_5_5 = np.array(ker_5_5)

    print(conv2(img,ker_3_3,'same'))
    print(conv2(img,ker_5_5,'same'))
