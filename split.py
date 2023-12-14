import splitfolders
splitfolders.ratio("C:/input/dataset/", output="C:/split/", seed=1234, ratio=(0.8,0.1,0.1))

#하고 valid로 이름 바꿀것!!!!

#모델 학습 시 데이터셋을 train, validation, test 데이터 셋으로 나눔
#train set: 학습에 사용되는 훈련용 데이터
#valid set: 모델의 일반화 능력을 높이기 위해 학습중에 평가에 사용되는 검증데이터
#test set: 학습 후에 모델의 성능을 평가하기 위해서 사용되는 테스트용 데이터

#train:valid:test = 8:1:1 비율로