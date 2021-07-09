# ImageProcessing

**Image Colorization**

**Main Objetive:** Image colorization (from Grayscale to RGB)

**Input Images:** As imagens de entrada serão em preto e branco. Selecionamos algumas fotos dessa base: https://www.flickr.com/photos/powerhouse_museum/albums (No known restrictions on publication.)

Exemplos:

<img src="bw_images/Miner.jpg" width="400"> <img src="bw_images/Saxophone.jpg" width="400">
<img src="bw_images/Break.jpg" width="400"> <img src="bw_images/Tabaco.jpg" width="400">


**Passos:**
-  Aplicar técnicas de enhancement nas fotos que forem necessárias.
   1. Aplicaremos a técnica de histograma cumulativo e histograma equalization.
   2. Utilizaremos a imagem resultante do passo anterior para colorir.
-  Colorir as imagens através de técnicas de CNN ou DeepLearning.
 

**Código inicial:** Projeto_final.py

**Resltado parcial:**
- Original:
   
   <img src="bw_images/The_fountain.jpg" width="700">
- Após enhancement:
   
   <img src="results_enhancement/The_fountain.jpg" width="700">
