# unittest

it should not bypass tests

```shell
pytest -s projects/cosmos/diffusion/v2/configs/i2v_wan/ --all
```


# porting and verify pretrained weights

* check `projects/cosmos/diffusion/v2/configs/i2v_wan/experiment/wan2pt1/resume_inference_test.py`

## Important notes

* for `14B + 720p` model, `wan2pt1_14B_res720p_16fps_Note-video_high_quality_v0`, the config  is not tested for training. May only works for inference.
* for `wan2pt1_14B_res720p_16fps_Note-video_high_quality_v0` and `wan2pt1_14B_res480p_16fps_Note-video_high_quality_v0`, the load path model weights only has regular model. No EMA weights!!!
