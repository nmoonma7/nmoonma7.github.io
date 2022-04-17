---
layout: post
title: interpret results in xgboost
description: >
  xgboost 모델의 결과를 해석해봅니다. tree를 그리고 그 결과를 이해하는 것과, xgboost에서의 feature importance를 

sitemap: false
hide_last_modified: true
categories:
    - notes
    - ml
---

# xgboost 결과 해석 

* toc
{:toc}

훈련시킨 xgboost 모델의 결과를 해석하기 위해서 트리를 그려보거나,feature importance를 구할 수가 있습니다. 

이번 포스트에서는 나타나는 트리의 순서가 정확히 어떻게 되는지, leaf value는 정확히 무엇을 말하는지, leaf value와 예측값은 어떻게 산출되는지 살펴보고자 합니다. 

또한 모델에서 feature의 영향력이 얼마인지 알기위해서 xgboost 라이브러리를 통해 feature importance를 구합니다. 여기서 feature importance가 정확히 어떻게 계산되는 것인지 알아보도록 하겠습니다.