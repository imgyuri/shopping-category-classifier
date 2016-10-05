### [shopping-category-classifier](https://github.com/im9uri/shopping-category-classifier)의 submission script

* 7기 SW마에스트로 과정 기초기술분야 과제(아 길다!)의 점수 1등 script입니다.

* Prediction score: 0.738979591837

* KoNLPy, Gensim(Word2Vec), Keras, CountVectorizer등을 이용해서 feature set을 만들었습니다.

* 모델은 xgboost, ExtraTreesClassifier, Logistic Regression등 다양한 시도를 해보았으나, LinearSVM이 가장 점수가 잘 나왔습니다.

* 실행에 필요한 library 설치 및 data를 넣어서 AWS community AMI로 만들었으므로, 사용하실분은 자유롭게 사용해주세요. (oregon region에 predict-shopping-category를 검색하시면 됩니다.)
