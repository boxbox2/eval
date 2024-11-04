``python train_old.py --custom_training_category --rotation_category hazelnut grid --gpu_id 3 --datasource eva``
``--custom_training_category``表示你想单独训练一个类别，是否进行旋转数据增强可看代码有```slight ，no，rotation``` 
不单独训练类别，直接``python train_old.py``


使用预训练模型推理``python eval_new.py --catagory screw``

[谷歌云盘](https://github.com/boxbox2/eval/tree/master )
