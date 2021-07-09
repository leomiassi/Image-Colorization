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

  prototxt = 'modelos/colorization_deploy_v2.prototxt'
  model = 'modelos/colorization_release_v2.caffemodel'
  points = 'modelos/pts_in_hull.npy'

  # load our serialized black and white colorizer model and cluster
  # center points from disk
  print("[INFO] loading model...")
  net = cv2.dnn.readNetFromCaffe(prototxt, model)
  pts = np.load(points)
  # add the cluster centers as 1x1 convolutions to the model
  class8 = net.getLayerId("class8_ab")
  conv8 = net.getLayerId("conv8_313_rh")
  pts = pts.transpose().reshape(2, 313, 1, 1)
  net.getLayer(class8).blobs = [pts.astype("float32")]
  net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
  # load the input image from disk, scale the pixel intensities to the
  # range [0, 1], and then convert the image from the BGR to Lab color
  # space
  #image = cv2.imread(image)
  scaled = image.astype("float32") / 255.0
  lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
  # resize the Lab image to 224x224 (the dimensions the colorization
  # network accepts), split channels, extract the 'L' channel, and then
  # perform mean centering
  resized = cv2.resize(lab, (224, 224))
  L = cv2.split(resized)[0]
  L -= 50
  # pass the L channel through the network which will *predict* the 'a'
  # and 'b' channel values
  'print("[INFO] colorizing image...")'
  net.setInput(cv2.dnn.blobFromImage(L))
  ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
  # resize the predicted 'ab' volume to the same dimensions as our
  # input image
  ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
  # grab the 'L' channel from the *original* input image (not the
  # resized one) and concatenate the original 'L' channel with the
  # predicted 'ab' channels
  L = cv2.split(lab)[0]
  colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
  # convert the output image from the Lab color space to RGB, then
  # clip any values that fall outside the range [0, 1]
  colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
  colorized = np.clip(colorized, 0, 1)
  # the current colorized image is represented as a floating point
  # data type in the range [0, 1] -- let's convert to an unsigned
  # 8-bit integer representation in the range [0, 255]
  colorized = (255 * colorized).astype("uint8")
  # show the original and output colorized images
  cv2.imshow("Original", image)
  cv2.imshow("Colorized", colorized)
  #cv2_imshow(image1)
  #cv2_imshow(image)
  #cv2_imshow(colorized)
  #cv2.waitKey(0)
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
