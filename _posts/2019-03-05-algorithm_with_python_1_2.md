---
layout: post
title: algorithm with python - hanoi tower (2)
categories: algorithm
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2019-03-05
tags: [algorithm][hanoi tower]
image:
  path: /images/abstract-3.jpg
  feature: abstract-3.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

### hanoi tower algorithm

```python
def move_disk(disk_num, start_peg, end_peg):
    print("%d번 원판을 %d번 기둥에서 %d번 기둥으로 이동" % (disk_num, start_peg, end_peg))

def hanoi(num_disks, start_peg, end_peg):
    if num_disks ==0:
        return
    hanoi(num_disks-1, start_peg,6-(start_peg+end_peg))
    move_disk(num_disks, start_peg, end_peg)
    hanoi(num_disks-1, 6-(start_peg+end_peg),end_peg)

# test
#hanoi(2, 1, 3)
hanoi(3, 1, 3)
```
