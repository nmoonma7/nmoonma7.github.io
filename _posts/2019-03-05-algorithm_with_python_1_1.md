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

{% highlight scss %}
.highlight {
  margin: 0;
  padding: 1em;
  font-family: $monospace;
  font-size: $type-size-7;
  line-height: 1.8;
}
{% endhighlight %}

```python
def is_palindrome(word):
    for i in range(len(word)//2):
        if word[i] != word[len(word)-(i+1)]:
            return False
	
    return True
    
# 테스트
print(is_palindrome("racecar"))
print(is_palindrome("stars"))
print(is_palindrome("토마토"))
print(is_palindrome("kayak"))
print(is_palindrome("hello"))
```
