# Inofficial steps to reproduce the results

* I tried many ways to reproduce the results and found that the following parameters work well.


## Deepfashion

* Deepfashion is trained on single images, so it is a *static* dataset.

```bash
export CUDA_VISIBLE_DEVICES=X # instead of main.py --gpu xx
python main.py baseline_deepfashion_256 \
--dataset deepfashion --mode train --bn 8 --static \
--in_dim 256 \
--reconstr_dim 256 \
--covariance \
--pad_size 25 \
--contrast_var 0.01 \
--brightness_var 0.01 \
--saturation_var 0.01 \
--hue_var 0.01 \
--adversarial \
--c_precision_trans 0.01
```

* Note that I had to make a custom split of the data for Deepfashion, which is basically going through all the data in the 
in-shop subset of Deepfashion and filter out those images where all keypoints are visible.
* To get the keypoints, I simply used Alpha Pose.
* The custom subset is released under [custum_datasets/deepfashion](custum_datasets/deepfashion/README.md)

