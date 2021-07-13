import numpy as np
import imageio as im
import math
import matplotlib.pyplot as plt
import cv2
import copy

#####################################################################################################################

def correcao_gamma(imagem, gamma):
  return np.floor(255 * (((imagem)/255.0)**(1/gamma)))

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
    
    N, M = image.shape[0], image.shape[1]# O tamanho da imagem
    
    image_eq = np.zeros([N,M]).astype(np.uint8)# Cria uma matriz vazia para armazenar a imagem equalizada
    
    for z in range(no_levels):# Para cada valor de intensidade transforma em uma nova intensidade
      s = ((no_levels-1)/float(M*N))*histC[z]# Calcula qual o valor s baseado no valor do histograma cujo valor é z
        
      image_eq[ np.where(image == z) ] = s# Quando a coordenada da matriz corresponder com o valor da imagem coloca o valor s
    
    return image_eq
  
#####################################################################################################################
  
def enhance_image1(image):
    img_new = copy.copy(image)

    image1 = image[:,:,0]
    image2 = image[:,:,1]
    image3 = image[:,:,2]
    img_new1 = histogram_equalization(image1, 256, histogram(image1, 256))
    img_new2 = histogram_equalization(image2, 256, histogram(image2, 256))
    img_new3 = histogram_equalization(image3, 256, histogram(image3, 256))

    img_new[:,:,0] = img_new1
    img_new[:,:,1] = img_new2
    img_new[:,:,2] = img_new3

    return img_new
  
#####################################################################################################################

def enhance_image2(image,gamma):
    img_new = copy.copy(image)

    image1 = image[:,:,0]
    image2 = image[:,:,1]
    image3 = image[:,:,2]
    img_new1 = correcao_gamma(image1, gamma).astype(np.uint8)
    img_new2 = correcao_gamma(image2, gamma).astype(np.uint8)
    img_new3 = correcao_gamma(image3, gamma).astype(np.uint8)

    img_new[:,:,0] = img_new1
    img_new[:,:,1] = img_new2
    img_new[:,:,2] = img_new3

    return img_new
  
##################################################################################################################### 

def coloring(image):

    # Carregando os modelos
    protoFile = 'modelos/colorization_deploy_v2.prototxt'
    weightsFile = 'modelos/Imagens/colorization_release_v2.caffemodel'
  
    # Carrega os pontos de cluster
    pts_in_hull = np.load('modelos/Imagens/pts_in_hull.npy')
  
    # Carrega a rede
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
  
    # Add o centro do cluster como 1x1 convolutions para o modelo
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]
  
    # Amostras do OpenCV
    W_in = 224
    H_in = 224
  
    img_rgb = (image[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
    # RGB em LAB
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_l = img_lab[:,:,0] # Pega o L
  
    # Faz um resize do L para o tamanho da rede de entrada
    img_l_rs = cv2.resize(img_l, (W_in, H_in))
    img_l_rs -= 50 # Subtrai 50 da média centralizada
  
    net.setInput(cv2.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0,:,:,:].transpose((1,2,0)) # O resultado
  
    (H_orig,W_orig) = img_rgb.shape[:2] # Tamanho original da imagem
    ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # Concatena com o L da imagem original
    img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)
  
    colorized = (img_bgr_out*255).astype(np.uint8)
    cv2_imshow(image)
    cv2_imshow(colorized)
    
    # Retornando a imagem colorida
    return colorized

##################################################################################################################### 
  
def main():
    #Carregando a imagem
    filename = input().rstrip()
    filename = "bw_images/" + filename #Adicionando o repositório no nome
    print(filename)
    image1 = cv2.imread(filename)
    
    #Aplicando o enchancement
    image = enhance_image1(image1)
    image = enhance_image2(image1)
    
    #Aplicando a colorizaçao
    image = coloring(image)
    
    #Plotando um comparativo entre a imagem de entrada e a imagem após o enhancement
    
    #Salvando a imagem gerada em um arquivo
    filename = re.split("\/", filename)[1] #Usando regex para extrair o nome da imagem.
    filename = "resultados/" + filename #Adicionando o repositório no nome
    im.imwrite(filename, image)  #Salvando a imagem
    
main()
