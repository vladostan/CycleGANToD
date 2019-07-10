#python3 train.py --dataroot ./datasets/day2night_inno --name day2night_inno_cyclegan --model cycle_gan
#python3 segmentation_train_linknet_resnet18.py
python3 train.py --dataroot ./datasets/icevision_day_and_night --name icevision_day_and_night_cyclegan --model cycle_gan
#python3 train.py --dataroot ./datasets/exp --name exp --model cycle_gan
#python3 train.py --dataroot ./datasets/day2night_inno --name day2night_inno_cyclegan2 --model cycle_gan
python3 train.py --dataroot ./datasets/day2night_bdd --name day2night_bdd_cyclegan --model cycle_gan
