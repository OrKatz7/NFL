# NFL 1st and Future - Impact Detection
This repository developed by [Or Katz](https://www.linkedin.com/in/or-katz-9ba885114/), [Usha Rengaraju](https://medium.com/@usharengaraju), [Shlomo Kashani](https://www.linkedin.com/in/quantscientist/), [Kumar Shubham](https://www.linkedin.com/in/kumar-shubham-iitd/) and [PRATEEK KUMAR AGNIHOTRI
](https://www.linkedin.com/in/prateek-kumar-agnihotri-18b498157/)  for the 17th place solution to the kaggle - NFL 1st and Future - Impact Detection

![alt text](https://github.com/OrKatz7/NFL/blob/main/docs/nfl.png)

# step 1 - make yolo data
## change param:
```
%cd yolov5
PATH_DF = '/data/nfl-impact-detection/train_labels.csv'
ALL_DATA = -1 #change to 0 if you want to train only images with impact
video_dir = '/data/nfl-impact-detection/train'
out_dir = 'train_images'
```
## Run:
```
%cd yolov5
python3 make_yolo_data.py
```
The images in yolo format save in: convertor/fold0/images/

# step 2 - train
```
%cd yolov5
sh weights/download_weights.sh
pip install -U seaborn
go to - models/yolov5x.yaml change nc to number of classes
go to - data/nfl.yaml change nc to number of classes, class name and image loc
```
## run
```
%cd yolov5
python3 train.py --img 1280 --batch 16 --epochs 10 --data data/nfl.yaml --cfg models/yolov5x.yaml --name yolov5x_fold0 --weights yolov5x.pt --device 0,1,2,3
```
# Submit
```
run_all - yolo-with-player-assignment-pp.ipynb
On kaggle - https://www.kaggle.com/ks2019/yolo-with-player-assignment-pp
```
# References
```
https://github.com/ultralytics/yolov5
```
# LICENSE
YoloV5 LICENSE - GNU General Public License v3.0 - https://github.com/ultralytics/yolov5/blob/master/LICENSE
