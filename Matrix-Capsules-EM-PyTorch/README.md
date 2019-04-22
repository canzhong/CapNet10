# Matrix Capsules with EM Routing using ResNeXt feature sampling
A Pytorch implementation variation of [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb) using 8 ResNeXt block as the feature map extractor instead of using the tradition CNN used in the base architecture.

Architecture :

ResNextB(in_channels=1, out_channels=X, stride=1, cardinality=2, base_width=16, widen_factor=2)
ResNextB(in_channels=X, out_channels=Y, stride=1, cardinality=1, base_width=16, widen_factor=2)
ResNextB(in_channels=Y, out_channels=X, stride=1, cardinality=2, base_width=16, widen_factor=2)
ResNextB(in_channels=X, out_channels=Y, stride=1, cardinality=1, base_width=16, widen_factor=2)
ResNextB(in_channels=Y, out_channels=X, stride=1, cardinality=2, base_width=16, widen_factor=2)
ResNextB(in_channels=X, out_channels=Y, stride=1, cardinality=1, base_width=16, widen_factor=2)
ResNextB(in_channels=Y, out_channels=X, stride=1, cardinality=2, base_width=16, widen_factor=2)
ResNextB(in_channels=X, out_channels=A, stride=1, cardinality=1, base_width=16, widen_factor=2)
PrimaryCaps(A, B, 1, P, stride=1)
ConvCaps(B, C, K, P, stride=2, iters=iters)
ConvCaps(C, D, K, P, stride=1, iters=iters)
ConvCaps(D, E, 1, P, stride=1, iters=iters, coor_add=True, w_shared=True)

X = 16, Y = 32, A=B=C=D = 16, D = 10

## CIFAR10 experiments

The experiments are conducted on Nvidia Tesla V100.
Specific setting is `lr=0.01`, `batch_size=16`, `weight_decay=0`, Adam optimizer

Following is the result after 40 epochs training:

| Arch | Iters | Coord Add | Loss | BN | Test Accuracy |
| ---- |:-----:|:---------:|:----:|:--:|:-------------:|
| A=B=C=D=16 | 3 | Y | Spread    | Y | 66.92 |

The training time of `A=B=C=D=16` for a 16 batch is around `0.0367s`.

## Reference
The research done is solely for non-profit academic purposes. Code from several repos were altered and pieced together to create an architecture compatible with the experiment hypothesis.

Capsule components were inspired from https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch
ResNeXt components were inspired from https://github.com/prlz77/ResNeXt.pytorch
