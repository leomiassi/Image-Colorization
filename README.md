# ImageProcessing

**Image Colorization**

**Main Objetive:** Image colorization (from Grayscale to RGB)

**Input Images:** As imagens de entrada serão em preto e branco. Selecionamos algumas fotos dessa base: https://www.flickr.com/photos/powerhouse_museum/albums (No known restrictions on publication.)

Exemplos:

<img src="bw_images/Miner.jpg" width="400"> <img src="bw_images/Saxophone.jpg" width="400">
<img src="bw_images/Break.jpg" width="400"> <img src="bw_images/Tabaco.jpg" width="400">


**Passos:**
-  Aplicar técnicas de enhancement nas fotos que forem necessárias.
   1. Aplicaremos a técnica de histograma cumulativo, histograma equalization e gamma enhancement.
   2. Utilizaremos a imagem resultante do passo anterior para colorir.
-  Colorir as imagens através de técnicas de CNN/DeepLearning.
   1. Converter a imagen de entrada para o Lab color space.
   2. Usar o canal L como entrada para a rede de treinamento tentar prever os canais 'a' e 'b'.
   3. Combinar o canal L de entrada com os caneis 'a' e 'b' que foram previstos.
   4. Converter a imagem do formato Lab para o formato RGB.
 

**Código inicial:** Projeto_final.py

**Resltados:**
- Original:
   
   <img src="bw_images/The_fountain.jpg" width="700">
- Histogram Enhancement:
   
   <img src="results_enhancement/The_fountain.jpg" width="700">
- Gamma Enhancement: 

   <img src="resultados/gamma_The_fountain.jpg" width="700">
- Colored:

   <img src="resultados/color_gamma_The_fountain.jpg" width="700">
