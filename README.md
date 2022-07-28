# VirtualTryOn

## Introduction
*Virtual try-on을 위한 preprocessing 자동화 및 고화질 VITON-HD Generative model 개발.* 
> 배경과 모델을 합성하기 위해 먼저 배경에서 인물이 들어갈 위치를 정하고 배경에 맞게 모델의 크기를 조절하는 부분은 수작업으로 진행하였다.
> 배경에서 인물이 들어갈 위치 좌표와 모델의 크기에 대한 좌표 3가지를 입력해 주면 자동적으로 모델의 크기 및 위치 변경을 통해 배경과 합성이 이루진다. 배경과 합성한 모델은 원래의 모델에서 새로운 의상의로 입힌 후 배경에 맞게 위치 및 크기를 조절하여 합성을 하였다.
#### Pre-Processing : step1 , step2, step3
   - Segmentation 네트워크를 이용하여 인물 영역 분리 및 배경 제거를 통해 가상 피팅 준비 작업 및 인물 영역 segmentation 과정 후 contour modification 처리 과정을 통해 인물의 경계 영역 보정 (**step1**)
   - Virtual try-on을 위해 모델에 대한 pose estimation을 통한 신체의 체형 및 자세를 추정 및 정보 분석(**step2**)
   -  Virtual try-on을 위해 모델에 대한 body part segmentation (Instance-level Human Parsing via Part Grouping Network)(**step3**)
#### Virtual Try-On : step4
   - VITON-HD를 참고하여 고화질 generative model 개발 
   - 모델과 상의 학습 데이터를 pair로 하여 상대적으로 고화질 virtual try-on 구현 및 빠른 학습 가능 (test_pairs.txt)
   - Inputs : 
      - 배경 제거된 모델 사진(inputs/image) 
      - Openpose 통한 신체의 체형 및 자세를 추정 및 정보(inputs/openpose-img 및 inputs/openpose-json)
      - Part Grouping Network(PGN) 통한 output/cihp_parsing 이미지
      - 상의 의상 이미지 및 mask 정보 (inputs/cloth 및 inputs/cloth_mask)
   - Output : 
      - 512 x 512 모델 virtual try-on 결과 이미지 (output/viton-hd) 
#### Post-Processing : step5
   - Virtual try-on 수행 후 배경 합성을 통한 자연스러운 이미지 생성
   - Inputs : backgrounds.txt
      - 합성할 배경 사진와 모델에 position 정보(inputs/bg_images)
   - Output :
      -  배경 합성을 통한 자연스러운 이미지 생성 (output/final)

## Environment Setup
*본 연구에서는 segmentation 네트워크, pose estimation 네트워크, Part Grouping Network(PGN) 네트워크, VITON-HD virtual try-on 네트워크, 네가지의 네트워크들를 사용한다. 각각 네트워크의 Environment가 다르며 동시에 실행하기 위해 environment setting을 진행해야한다.*

0. weights 파일 다운로드 링크 
> (download_weights.txt 파일 참고) : 
 <https://drive.google.com/file/d/1Cp-JBBpOmvQp8nttmInmr-hNyksFgXAC/view?usp=sharing>
1. 배경을 제거하기 위한 segmentation network: AdelaiDet's Networks
> AdelaiDet Networks 중에 가장 좋은 성능을 보여준 CondInst와 BlendMask를 사용한다. (학습 weights 파일은 coco weights 파일을 사용했으며, 다운로드하는 방법은 download_weights.txt 파일 참고)
> >성능 결과 : <https://www.notion.so/Results-for-AdelaiDet-s-Networks-07718ebe25634853afb56da20ec34a09>
> 
> Adelai Networks 위한 환경 세팅 :
> > ```
> > cd background 
> > conda env create -f environment_adelai.yaml
> > ```
> 

2. 신체의 체형 및 자세를 위한 Pose estimation Network: OpenPose
> OpenPose Network는 cmake를 통해 build 해야한다. Build하는 방법은 대한 자세한 내용은 openpose/BuildOpenPose.md 파일 참고 
> 
> reference : <https://github.com/CMU-Perceptual-Computing-Lab/openpose>

3. Part Grouping Network(PGN) : <https://github.com/Engineering-Course/CIHP_PGN>
> PGN is a state-of-art deep learning methord for semantic part segmentation, instance-aware edge detection and instance-level human parsing built on top of Tensorflow. (학습 weights 파일은 Crowd Instance-level Human Parsing (CIHP) Dataset를 사용했으며 다운로드하는 방법은 download_weights.txt 파일 참고)
> 
> CIHP_PGN 위한 환경 세팅 :
> > ```
> > cd CIHP_PGN 
> > conda env create -f environment_CIHP_PGN.yaml
> > ```
> 

4. Virtual try-on 수행하기 위한 VITON-HD network
>  VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization : <https://arxiv.org/abs/2103.16874>
>  
> AI Hub 에서 8 만개의 공개 데이터셋을 다운받아 가공하여 사용하였으며 본 연구에 맞는 데이터를 선별하여 15,000 개의 모델과 의상 이미지를 학습 데이터
로 사용하였다 (<https://aihub.or.kr/aidata/30755>) 
>
> 학습 weights 파일을 다운로드하는 방법은 download_weights.txt 파일 참고
> 
> VITON-HD 위한 환경 세팅 :
> > ```
> > cd VITON-HD 
> > conda env create -f environment_VITON_HD.yaml
> > ```
> 

## Inference

*중 4가지 network 환경 세팅을 완료한 후 아래와 같이 inference 과정을 진행한다*  
*(test.py option에서 각 network 환경에 python 경로 설정 필요)*

> ### Step1
> *input image에서 배경을 삭제하는 단계*
> > Inputs :   
> > Copy inference images into **inputs/org_img** folder
>    
> > Run :   
> > ```
> > python test.py --step step1
> > ```
>  
> > Results :   
> > **output/background_step1** 에 생성
   
> ### Step2
> *image resizing 및 openpose data를 생성하는 단계*   
> *VITON-HD network 위해 input size를 resizing 및 cropping 한 후 신체의 체형 및 자세를 위한 openpose 실행*
>   
> > Inputs : 
> >
> > step1에서 **output/background_step1/** 에 생성된 배경 제고된 이미지.
>    
> > Run :   
> > ```
> > python test.py --step step2 
> > ```
>  
> > Results :   
> > **inputs/image** 에 중점 이동 후 size 조정한 이미지 생성   
> > **inputs/openpose-img** 에 openpose 이미지 생성   
> > **inputs/openpose-json** 에 openpose json data 생성
>
> >  추가적으로 VITON-HD network를 실행하기 위해 필요한 inputs/test_pairs.txt 파일 자동 생성. (inputs/gen_val_list.py 파일 참고)
> >

> ### Step3
> *Part Grouping Network(PGN)을 통해 cihp_parsing 이미지 생성*
> > Inputs :   
> > step2 에서 생성된 **inputs/image** 이미지, 
> >  
> > **inputs/val.txt** 에 generation할 image 와 label 경로 입력 (label은 고정값을 넣어주면 됨)   
> > 
> > **inputs/val_id.txt** 에 input edge 경로 입력 (고정값을 넣어주면 됨, 다만 image 개수와 일치)  
> >  
> > **inputs/palette_ref.png** (palette 참고용)  
> >
> > 추가적으로 **inputs/labels**와 **inputs/edges** 에 **512x512.png** 필요  
>    
> > Run :   
> > ```
> > python test.py --step step3
> > ```
>  
> > Results :   
> > **output/cihp_parsing** 에 결과 이미지 생성   
>  


> ### Step4
> *cloth synthesis[VITON-HD]*
> > Inputs :   
> > step2 에서 생성된 **inputs/openpose-img** 와 **inputs/openpose-json**    
> > 합성할 cloth image와 해당 mask 이미지 -  **inputs/cloth**  ,  **inputs/cloth-mask**   
> > 합성할 model과 cloth의 pair - **inputs/test_pairs.txt** (step2에서 자동 생성됨)
>    
> > Run :   
> > ```
> > python test.py --step step4
> > ```
>  
> > Results :   
> > **output/viton-hd** 에 결과 이미지 생성   

> ### Step5
> *배경 이미지 합성*
> > Inputs :   
> > **inputs/bg_images** folder에 합성할 background image와 model position info txt 파일   
> > **inputs/backgrounds.txt** 에 합성할 background과 position txt file path 작성  
>    
> > Run :   
> > ```
> > python test.py --step step5
> > ```
>  
> > Results :   
> > **output/final** 에 배경 이미지 결과 이미지 생성   
>

***
## VITON-HD virtual try-on network 학습
*기본 경로 : VITON-HD*  
Conda Env Activation :
```
cd VITON-HD
conda activate VITON_HD
```
   
*Preprocessing*
1. cloth와 cloth-mask 이미지 size 조정 (512x512)   
2. model image를 중점으로 이동시키고 size 조정 (512x512)   
3. 2에서 만든 image로 parse map과(segment) openpose 데이터 추출   
4. 학습할 model과 cloth pair를 **datasets/train_pairs.txt**에 작성   
   

*학습을 이어서 할 때는 --continue_train 옵션 사용*
*GPU가 여러개인 환경에서 학습을 하고자 할 때에는 CUDA_VISIBLE_DEVICES=[사용하고자 하는 GPU] 를 커맨드 앞에 
   
> ##### Seg Generation
> > Inputs :   
> > **datasets/train/cloth** folder에 학습할 cloth copy   
> > **datasets/train/cloth-mask** folder에 학습할 cloth mask copy   
> > **datasets/train/image-parse** folder에 에 parse map(segment) copy   
> > **datasets/train/openpose-img** folder에 에 openpose image copy   
> > **datasets/train/openpose-json** folder에 에 openpose json copy   
>    
> > Run :   
> > ```
> > python train.py --train_model seg --name [트레이닝 이름]
> > ```
>  
> > Results :   
> > **checkpoints/[테스트 이름]** 에 생성
   
> ##### Gmm Generation
> > Inputs :   
> > **datasets/train/image** folder에 학습할 model image copy
>    
> > Run :   
> > ```
> > python train.py --train_model gmm --name [트레이닝 이름]
> > ```
>  
> > Results :   
> > **checkpoints/[테스트 이름]** 에 생성

   
> ##### Alias Generation
> > Inputs :   
> > GMM 학습 결과로 만들어낸 warped cloth를 **datasets/train/warped_cloth** folder에 copy   
> > GMM 학습 결과로 만들어낸 warped cloth-mask를 **datasets/train/warped_cloth-mask** folder에 copy   
>    
> > Run :   
> > ```
> > python train.py --train_model alias --name [트레이닝 이름]
> > ```
>  
> > Results :   
> > **checkpoints/[테스트 이름]** 에 생성
   
***
## Q&A
> **Q1) 개발 코드 공유-github (readme/주석 포함)**  
>
> => 아래와 같이 기존에 공유드렸었던 github에 코드 및 readme 주석을 추가 하였음
> 
> https://github.com/ssai-snu/gs_viton
>
> **Q2) (d) netwok는 6차까지 training 진행했는데, 어떤 것을 바꿔가며(왜 바꾸었는지) training했는지 기술 (가능하면 네트워크, 학습epoch, 시간 포함)**
> 
> => 실험한 네트웍은 5차까지 training을 진행하였으며 자세한 설명은 아래와 같음
> 
> > 1) VITON-HD 네트워크 학습 epoch 및 학습 시간 
> > 
> > 4개의 블록으로 이루어진 VITON-HD 각각의 학습 시간은 아래와 같다. 
> > 
> > (a) Pre-processing 블록은 학습이 따로 필요하지 않음
> > 
> > (b) Segment Generation 블록 학습 : Paired Image 를 사용하여 학습 시간 및 성능 개선
> > 
> > 200epoch 약 60 시간 소요 (A100 1 대 기준 )
> > 
> > (c) Clothes Deformation 블록 학습 (GMM network) : Second order difference constraint 를 적용하여 cloth wrapping
> > 
> > 200epoch 약 80 시간 소요 (A100 1 대 기준 )
> > 
> > (d) Alias Generator 블록 학습 : 모델에 새로운 의상 가상 피팅, 모델의 자세 및 신체 분위 검출
> >
> > 200epoch 약 80 시간 소요 (A100 1 대 기준 )
> > 
> > 2) Alias Generator 최대 5차까지만 학습 하였으며, 3차, 4차, 5차 학습의 차이점은 파라미터 외에는 큰 차이가 없음
> > 
> > 4차에서만 GAN Loss 연산에서 target_is_real param에 False이고, 3차와 5차일 때 True를 넣어서 실험하였다는 차이가 있었고, 결과적으로 True를 넣은 경우에서 좋은 결과를 얻음, 그리고 3차와 5차의 차이점은 GAN feature matching loss의 lambda 값의 차이다(VITON HD 논문 참조 함). (3차 : 5, 5차 : 10)
> 
> **Q3) (그 외 네트워크 학습시 시도했던 방법들...그리고 실제로 시도했던 시행착오 . 예를들면 second-order문제라든지, end to end network의 어려웠던점.. 시간이 없어서 못했던점 등은 없었는지)**
> 
> => VITON, CP-VTON, O-VITON 등의 다양한 네트웍을 시행했으며, VITON, CP-VTON 네트웍은 저해상도(256 x 192)만 지원하며, 상의만 try-on 할수 있다는 단점이 있으며, O-VITON은 고해상도와 하의까지 try-on 할수 있다는 장점이 있지만, 코드 및 데이터셋이 비공개이며 unpair 방식이기 때문에 이미지 퀄리티가 무척 낮은편이다. 또한 상대적으로 학습시간이 무척 오래 걸린다는 단점이 있었다. 
> 
> **Q4) future work에 대해서 구체적으로 고쳐야 할 부분에 대한 의견 (하의부분을 하려면 어디어디를 손봐야하는지, 각 task 개선시 좀더 구체적인 아이디어는?)**
> 
> => 향후 future work로는 다음의 개선점이 있음
> 
> > 1. 학습데이터 추가 : 기존 논문과 같이 1만5천개에서 5만개의 학습데이터를 충분히 추가하여 학습한다면 고해상도 이미지 및 고화질의 texture를 생성할 수 있을것으로 기대함
> > 
> > 2. 하의 try-on 적용 : 하의 부분 적용을 위한 내용은 아래와 같음
> > 
> > *train_dataset.py & test_dataset.py*
> > 
> > > get_parse_agnostic() 에서 masking하는 부분을 상의에서 하의로 수정
> > > 
> > > get_img_agnostic() 에서 masking하는 부분을 상의에서 하의로 수정
> > 
> > *VitonHDModel.py*
> > 
> > > GmmModel() class의 forward()에서 parse_map[] 곱셈 연산하는 부분 상의에서 하의로 parameter 변경
> > > 
> > > AliasModel() class의 forward()에서 alias_parse[] 곱셈 연산하는 부분 상의에서 하의로 parameter 변경
> > 
> > 
> > 3. 다양한 자세에 대한 학습데이터 추가 : 정면의 바른 자세뿐만 아니라 다양한 자세도 학습 데이터로 추가한다면 model의 다양한 자세에 대해서도 자연스럽게 try-on 시킬 수 있기 때문에 훨씬 좋은 성능이 나올것으로 기대함
> 
> **Q5) 추가 데이터 set 확보전략 (포즈, 모델 다양성, 복종, resolution 등 좀 더 구체적인 정보 포함)**
> 
> => 정면의 바른 자세로 구성된 데이터셋에서 팔을 들고 있는 자세, 팔이 꼬인 자세 및 비스틈한 자세 등 피팅 모델이 표현하는 복잡한 자세를 가진 이미지에 대한 데이터셋을 확보 할 예정이며 의상에서는 피부가 많이 노출되는 반팔셔츠, 나시, 수영복 등의 다양한 데이터셋과 나라별 전통의상 등의 독특한 의상 등도 추가하면 보다 좋은 이미지를 얻을 있을것 같음
> 
> **Q6) 평가 방식을 (FID, IS, SIMM 등) 기존 viton-hd 논문과 비교가 가능한지**
> 
> => 본 기술은 VITON-HD 논문을 기반으로 개발하였기 때문에 알고리즘 관점에서 보면 비슷한 결과가 나올것으로 예상 되지만 학습 데이터는 VITON-HD 논문에서 사용했었던  데이터가 아닌 AI-HUB 공개 데이터셋을 사용했기 때문에 데이터와 데이터수 등에서 차이가 발생할수도 있음
> 
> **Q7) viton-hd에 사용된 기술(라이브러리 등)에 대한 상업적 사용이 문제가 없는지 (라이선스 조사 필요)**
> 
> => VITON-HD github에서 라이선스 관련 내용은 아래와 같음
> 
> > License
> >
> > All material is made available under Creative Commons BY-NC 4.0 license by NeStyle Inc. You can use, redistribute, and adapt the material for non-commercial purposes, as long as you give appropriate credit by citing our paper and indicate any changes that you've made.

> **Q8) 사람 crop후 경계면에 대한 보정 로직에 대한 설명(수식 포함), 잘안되는 예제 포함하여 향후 개선 포인트 기술도 가능한지**
> 
> => 배경 제거 하기 위해 Adelai segmantation Network를 사용한 후 추가 영역 보정 작업을 하였으며 배경 제거시 경계면 근처에서 이전 배경이 나타남.
> 
> 경계면 부분의 pixel 주변 값에 대해 탐색 및 보정 작업을 진행
> 
> 로직에 대한 설명(수식 포함) 및 예제는 다음의 링크에 자세히 설명 : https://highfalutin-parka-155.notion.site/Remove-Background-44bde2dc55e646e78441cf0cc1feb1d1

   ***
