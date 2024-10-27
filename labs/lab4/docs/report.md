# Lab 4: Monte Carlo Method

:man_student: Charles

## 实验概述

基于蒙特卡罗方法（Monte Carlo Method），近似计算圆周率的大小。

## 算法设计

随机生成大量横纵坐标均在 $(0, 1)$ 范围内的点，这些点的总数 $N_p$ 可以代表边长为 $1$ 的正方形的面积 $S_p$ ；其中，落在单位圆中的点的数量 $N_c$ 则可以代表该正方形中单位扇形的面积 $S_c$ 。

因为

$$
S_p = 1^2 = 1, \\
S_c = \frac{1}{4} \pi \times 1^2 = \frac{\pi}{4}
$$

且

$$
\frac{S_p}{S_c} = \frac{N_p}{N_c}
$$

所以整理可得

$$
\pi = \frac{4 N_c}{N_p}
$$

对应写出python代码如下：

```python
import random
import math

def MonteCarlo_Pi(point_num):
    """ 基于Monte Carlo Method计算圆周率 
    Args:
        point_num (int): 随机生成点的数目
    Returns: 
        float: 计算出的圆周率
    """
    if point_num <= 0:
        raise ValueError("参数（生成点的数目）必须为正数！")
    
    in_circle_num = 0  # 落在单位圆中的点的数量
    for _ in range(point_num):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if math.sqrt(x**2 + y**2) <= 1:
            in_circle_num += 1  # 点(x, y)在单位圆内
            
    return 4 * (in_circle_num / point_num)
```

## 实验结果

如下图所示，该算法成功求出了圆周率近似值，且生成的随机点越多，计算值越接近真实的 $\pi$ ：

![output1](./images/output1)

![output2](./images/output2)

