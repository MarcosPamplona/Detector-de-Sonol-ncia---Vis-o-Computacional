# Detector-de-Sonol-ncia---Vis-o-Computacional
Sistema de detecção de sonolência em tempo real utilizando visão computacional e webcam, desenvolvido em Python com OpenCV e MediaPipe.

O projeto utiliza o modelo Face Mesh para mapear pontos faciais e calcula o EAR (Eye Aspect Ratio) para identificar quando os olhos permanecem fechados por um período prolongado, emitindo um alerta visual de possível sonolência.

O sistema captura vídeo em tempo real pela webcam, detecta e rastreia landmarks faciais, calcula o EAR (Eye Aspect Ratio), monitora o tempo de olhos fechados e exibe alerta visual após 2 segundos de olhos fechados.

Bibliotecas Utilizadas
OpenCV
MediaPipe
NumPy
