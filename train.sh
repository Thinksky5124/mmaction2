###
 # @Author: Thyssen Wen
 # @Date: 2021-09-15 16:01:01
 # @LastEditors: Thyssen Wen
 # @LastEditTime: 2021-09-26 16:15:39
 # @Description: file content
 # @FilePath: /mmaction2/train.sh
### 
# 单GPU训练
# CUDA_VISIBLE_DEVICES=0 python3 ./tools/train.py configs/localization/WTAL/ACMNet_2x8_10e_AtivityNet_I3DACMNetfeature.py --validate --seed 0 --deterministic
# CUDA_VISIBLE_DEVICES=0 python3 ./tools/train.py configs/localization/WTAL/ACMNet_2x8_10e_thumos14_I3DACMNetfeature.py --validate --seed 0 --deterministic
# CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py configs/localization/WTAL/CoLA_1x8_10e_thumos14_I3DACMNetfeature.py --validate
# CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py configs/localization/WTAL/CoLA_1x8_10e_thumos14_I3DACMNetfeature.py --validate --seed 0 --deterministic
# CUDA_VISIBLE_DEVICES=2 python3 ./tools/train.py configs/localization/WTAL/CoLA_1x8_10e_AtivityNet_I3DACMNetfeature.py --validate --seed 0 --deterministic
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py --validate

# 多GPU训练
# ./tools/dist_train.sh configs/localization/WTAL/ACMNet_2x8_10e_AtivityNet_I3DACMNetfeature.py 4 --validate --seed 0 --deterministic