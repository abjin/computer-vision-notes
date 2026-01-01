# Computer Vision Notes

CNN 기반 컴퓨터 비전 아키텍처와 주요 비전 작업에 대한 학습 노트입니다.

> 이 노트들은 Velog 에서도 확인할 수 있습니다: https://velog.io/@abjin/series/cv

## Contents

### CNN Basics
기본 합성곱 연산과 풀링, GAP 등 핵심 개념을 다룹니다.

- [01. 합성곱/풀링/GAP](computer-vision/01-cnn-basic.md)
  - 합성곱, 스트라이드, 패딩
  - 풀링(평균/최대)과 이동 불변성
  - Global Average Pooling(GAP) 개념과 동기

### Architectures
고전부터 현대까지 대표적인 비전 모델과 학습 안정화/효율화 기법을 정리합니다.

- [02. LeNet-5](computer-vision/02-lenet5.md)
  - 역사적 중요성, C1–S2 구조
  - 공유 가중치·지역 연결·서브샘플링

- [03. AlexNet](computer-vision/03-alexnet.md)
  - AlexNet 아키텍쳐
  - GPU 병렬, ReLU, LRN, 중첩 맥스 풀링

- [04. VGG](computer-vision/04-vgg.md)
  - 3×3 컨볼루션 쌓기, 드롭아웃
  - 블록 구조와 FC 구성

- [05. ResNet](computer-vision/05-resnet.md)
  - 잔차 학습과 스킵 연결
  - 기울기 소실/성능 저하 문제 해결

- [06. Batch Normalization](computer-vision/06-batch-normalization.md)
  - 내부 공변량 이동, 정규화 수식/역전파
  - 안정적 학습과 수렴 가속

- [07. MobileNet](computer-vision/07-mobilenet.md)
  - Depthwise Separable Convolution
  - FLOPs/파라미터 절감

- [08. EfficientNet](computer-vision/08-efficientnet.md)
  - 복합 스케일링(깊이/너비/해상도)
  - MBConv, SE, Swish

- [09. Vision Tasks](computer-vision/09-vision-tasks.md)
  - 분류/탐지/분할 개요
  - 이동 불변성 vs 가변성

- [10. U-Net](computer-vision/10-unet.md)
  - 인코더–디코더, 스킵 연결
  - 세그멘테이션용 밀집 예측

- [11. Vision Transformer (ViT)](computer-vision/11-vit.md)
  - 패치 임베딩, 위치 임베딩, CLS 토큰
  - Transformer 인코더 블록 구성

### Experiments

- [LeNet-5 MNIST](experiments/lenet5_mnist.ipynb) - LeNet-5 구현 및 MNIST 분류
- [VGG16 CIFAR-10](experiments/vgg16_cifar10.ipynb) - VGG16 구현 및 CIFAR-10 분류