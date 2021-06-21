import numpy as np
import imageio as im
import math
import matplotlib.pyplot as plt


#####################################################################################################################

def histogram(image, no_levels):

  hist = np.zeros(no_levels).astype(int)# Cria um histograma com o tamanho da quantidade de tons de cinza que no caso é 255

  for i in range(no_levels):# Para todos os tons de cinza
    pixels_value_i = np.sum(image == i)# Somatorio das posições cujo valor do pixel da imagem corresponde ao valor i
    hist[i] = pixels_value_i# Armazena no histograma


  histC = np.zeros(no_levels).astype(int)# Cria um novo histograma cumulativo

  # computes the cumulative histogram
  histC[0] = hist[0] # O primeiro valor é corresponde com o primeiro valor do histgrama
  for i in range(1,  no_levels):#Valores de 1 até no_levels que no caso é 255
    histC[i] = hist[i] + histC[i-1]# Para cada intensidade acumula com o valor anterior para obter o histograma
            
  return histC

#####################################################################################################################

def histogram_equalization(image, no_levels, histC):
    
    N, M = image.shape# O tamanho da imagem
    
    image_eq = np.zeros([N,M]).astype(np.uint8)# Cria uma matriz vazia para armazenar a imagem equalizada
    
    for z in range(no_levels):# Para cada valor de intensidade transforma em uma nova intensidade
      s = ((no_levels-1)/float(M*N))*histC[z]# Calcula qual o valor s baseado no valor do histograma cujo valor é z
        
      image_eq[ np.where(image == z) ] = s# Quando a coordenada da matriz corresponder com o valor da imagem coloca o valor s
    
    return image_eq
  
#####################################################################################################################
  
def enhance_image(image):
    img_new = histogram_equalization(image, 256, histogram(image, 256))
    return img_new
  
#####################################################################################################################
  
def main():
    filename = input().rstrip()
    image = im.imread(filename)
    img_apri = enhance_image(image)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.show()
    plt.imshow(img_apri, cmap="gray", vmin=0, vmax=255)
    
main()
