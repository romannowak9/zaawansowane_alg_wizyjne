import numpy as np
import matplotlib.pyplot as plt
import cv2


def optical_flow(I, J, dW, dX, dY):
    YY, XX = J.shape
    u = np.zeros((YY, XX))
    v = np.zeros((YY, XX))

    for y in range(dW, YY - dW):
        for x in range(dW, XX  - dW):
            IO = np.float32(I[y-dW:y+dW+1, x-dW:x+dW+1])
            min_dist = np.inf
            min_x, min_y = 0, 0

            for surr_x in range(-dX, dX + 1):
                for surr_y in range(-dY, dY + 1):
                    if (x + surr_x - dW >= 0 and x + surr_x + dW < XX and
                        y + surr_y - dW >= 0 and y + surr_y + dW < YY):
                        
                        JO = np.float32(J[y+surr_y-dW:y+surr_y+dW+1, x+surr_x-dW:x+surr_x+dW+1])
                        distance = np.sum(np.sqrt((np.square(JO - IO))))

                        if distance < min_dist:
                            min_dist = distance
                            min_x, min_y = surr_x, surr_y

            u[y, x] = min_x
            v[y, x] = min_y
            
    return u, v


def vis_of(u, v, title):   
    magnitude, angle = cv2.cartToPolar(u, v)
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = angle * 90 / np.pi  
    hsv[:, :, 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  
    hsv[:, :, 2] = 255  

    # Konwersja HSV -> RGB i wyświetlenie obrazu
    img_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(title)
    plt.show()


def pyramid (im , max_scale):
    images = [im]
    for k in range (1 , max_scale):
        images.append(cv2.resize(images[k -1], (0 ,0),fx =0.5 ,fy =0.5))
    return images
