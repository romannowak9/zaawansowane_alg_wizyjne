import cv2
import matplotlib.pyplot as plt
import numpy as np

def optical_flow(I, J, dW, dX, dY):
    YY, XX = I.shape
    u = np.zeros((YY, XX))
    v = np.zeros((YY, XX))

    for j in range(dW, YY - dW):
        for i in range(dW, XX  - dW):
            IO = np.float32(I[j-dW:j+dW+1, i-dW:i+dW+1])
            min_distance = np.inf
            best_iW, best_jW = 0, 0

            for iW in range(-dX, dX + 1):
                for jW in range(-dY, dY + 1):
                    if (i + iW - dW >= 0 and i + iW + dW < XX and
                        j + jW - dW >= 0 and j + jW + dW < YY):
                        
                        JO = np.float32(J[j+jW-dW:j+jW+dW+1, i+iW-dW:i+iW+dW+1])
                        # distance = np.sqrt(np.sum((np.square(JO-IO))))
                        distance = np.sum(np.sqrt((np.square(JO - IO))))
                        #distance = np.linalg.norm(JO - IO) #odziwo wyniki inne niż w skrypcie XD

                        if distance < min_distance:
                            min_distance = distance
                            best_iW, best_jW = iW, jW

            u[j, i] = best_iW
            v[j, i] = best_jW

    return u, v
                            
                        
                


if __name__ == "__main__":
    #wczytanie obrazkow w skali szarosci
    I = cv2.imread("I.jpg")
    J = cv2.imread("J.jpg")
    
    # I = cv2.imread("cm1.png")
    # J = cv2.imread("cm2.png")
    
    
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
    
    # cv2.imshow("I", I), cv2.imshow("J", J), cv2.waitKey(0)
    
    differentialImage = cv2.absdiff(I, J)
    
    dW = 5
    dX = dY = 5
    YY, XX = I.shape
    # Zmniejszenie rozmiaru obrazów dla szybkości obliczeń
    scale = 0.5
    I = cv2.resize(I, (0, 0), fx=scale, fy=scale, interpolation= cv2.INTER_AREA)
    J = cv2.resize(J, (0, 0), fx=scale, fy=scale, interpolation= cv2.INTER_AREA)
 
    print(I.shape)
    # cv2.imshow("diff", differentialImage), cv2.waitKey(0)
    
    
                
    # Obliczenie przepływu optycznego
    u, v = optical_flow(I, J, dW, dX, dY)

    # Wizualizacja przepływu optycznego jako kolorowego obrazu HSV
    magnitude, angle = cv2.cartToPolar(u, v)
    hsv = np.zeros((I.shape[0], I.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = angle * 90 / np.pi  
    hsv[:, :, 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  
    hsv[:, :, 2] = 255  

    # Konwersja HSV -> RGB i wyświetlenie obrazu
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(flow_rgb)
    plt.axis("off")
    plt.show()
