###
 # @Author: Thyssen Wen
 # @Date: 2021-09-23 19:47:09
 # @LastEditors: Thyssen Wen
 # @LastEditTime: 2021-09-23 20:54:53
 # @Description: file content
 # @FilePath: /mmaction2/test.sh
### 
CUDA_VISIBLE_DEVICES=2 python tools/test.py work_dirs/ACMNet_2x8_10e_AtivityNet_I3DACMNetfeature/ACMNet_2x8_10e_AtivityNet_I3DACMNetfeature.py work_dirs/ACMNet_2x8_10e_AtivityNet_I3DACMNetfeature/best_mAP@0.50_epoch_170.pth --eval AR@AN