---
layout: post
title: xgboost
description: >
  Howdy! This is an example blog post that shows several types of HTML content supported in this theme.
sitemap: false
hide_last_modified: true
categories:
    - notes
    - ml
---

Regression  
Loss function  
$$\displaystyle\sum_{i=1}^{n}L(y_i , p_i) = \displaystyle\sum_{i=1}^{n}\frac{1}{2}(y_i -p_i)^2 $$

$y_i$는 관측값, $p_i$는 예측값, $n$은 관측 갯수  

Classification  
$\displaystyle\sum_{i=1}^{n}L(y_i , p_i) = \displaystyle\sum_{i=1}^{n}-[y_ilog(p_i)+(1-y_i)log(1-p_i)] $ 

xgboost는 다음과 같은 식을 최소화 하는 트리를 만든다  
$[\displaystyle\sum_{i=1}^{n}L(y_i , p_i)]+\frac{1}{2}\lambda O^2_{value} $ 

참고로 rT 도 식에는 들어가지만 트리를 만들고 최종적으로 가지치는 것에 이용되므로 우선 뺸다

