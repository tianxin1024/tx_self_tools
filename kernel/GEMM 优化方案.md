# GEMM 优化方案

### 1. 背景

Ascend310P 

**Multi-Core** (**AI CPU** + **Control CPU**): 昇腾 AI 处理器集成了**多个 CPU 核心**, 每个核心都有**独立的 L1 和 L2 Cache**, 所有核心**共享一个片上 L3 Cache**

![image-20250824051202708](/Users/tianxin/Library/Application Support/typora-user-images/image-20250824051202708.png)



硬件Atals 300I Pro 相关资料

https://support.huawei.com/enterprise/zh/doc/EDOC1100201570/d093df17?idPath=23710424|251366513|22892968|252309139|251052354

![image-20250824053051147](/Users/tianxin/Library/Application Support/typora-user-images/image-20250824053051147.png)



Atlas 300I Pro推理卡中的CPU是**华为自研的8核ARM架构核心**，主频1.9GHz。它与其他专用加速单元（NPU、编解码器等）协同，共同构成了一个高效的数据中心级AI推理解决方案。







![image-20250824080430952](/Users/tianxin/Library/Application Support/typora-user-images/image-20250824080430952.png)

**

论文的`5.6 Discussion`选择最优解：

1. 倒数第二、第三两种拆法直接被淘汰，因为需要完整的一行乘以一列，`MNK`值较大时需要很大的 cache，CPU cache并不大；
2. 因`Fig.8`和`Fig.9`在第四列都拆成了`GEBP`形式，所以拿它俩比较。由于在循环外 l1cache 级别 unpack C 远复杂于在循环内 register 级别 unpack C，所以`Fig.9`被淘汰；
3. `Fig.11`劣于`Fig.10`的理由也是外层 unpack C更复杂；
4. 最终剩下`Fig.8`和`Fig.10`。`Fig.8`在第四列处理竖着的小`slice`时，列主序是内存连续的，行主序不连续。`Fig.8`更适合列主序。

### 因此最终结论是：列主序用`Fig.8`最优；行主序用`Fig.10`最优。



前 3 个前提不考虑 TLB，假设只有 内存、cache 和 ALU ：

1. mc * kc 要小，小到 『 A + B的 nr 列 + C 的 nr 列 』能够一起塞进 cache
2. 如果 1. 被满足，CPU 计算时不再受内存速度的限制，即得到的`gflops`值就是真实的计算能力
3. A 或 A 的分块只会被加载进 Cache 一次，`gemm`过程中不会被换入又换出

后 2 个要考虑 TLB，因为 TLB miss 会 stall CPU：

4. mc 和 kc 要小，小到 『 A + B的 nr 列 + C 的 nr 列 』能够被 TLB 索引，即一定是小于 L2 cache 的。
5. A 或 A 的分块只被加载到 L2 cache 一次

因为`Fig.8`用的就是`GEBP`，所以想要高`gflops`就得满足上面 5 个条件。落到实处上就是一些参数限制，这些限制也是 `OpenBLAS level3.c`循环里写一堆`if-else`的理论根源：

1. mc ≈ kc
2. nr ≥ (Rcomp / 2 / Rload)，其中 Rcomp 是算力、Rload 是 L2 cache 到 register 的带宽
3. mc * kc ≤ K
4. mc * kc 只能占 cache 的一半

