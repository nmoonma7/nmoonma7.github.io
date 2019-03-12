---
layout: post
title: algorithm with python 1
categories: algorithm
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2019-03-05
tags: [algorithm]
image:
  path: /images/abstract-3.jpg
  feature: abstract-3.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

### ~~

```python
def is_palindrome(word):
    for i in range(len(word)//2):
        if word[i] != word[len(word)-(i+1)]:
            return False
	
    return True
    
# 테스트
print(is_palindrome("거꾸로꾸거"))
print(is_palindrome("geeks"))
print(is_palindrome("abcdcba"))
```
