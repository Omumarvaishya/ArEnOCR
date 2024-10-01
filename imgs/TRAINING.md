### For understanding training parameters and how to train and finetune
##### Recogntion
- [Read the training guidelines from scratch for rec given by PaddleOCR team](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/recognition_en.md)  
##### Detection
- [Read the training guidelines from scratch for det given by PaddleOCR team](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/detection_en.md)
##### Finetuning
- [Read the finetuning guidelines for both rec and det given by PaddleOCR team](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/finetune_en.md)
##### Config file
- [Read the meaning of all paramters present in the yml file for training](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/config_en.md)


### Setting Up
```
cd tabmind/
python3 -m venv venv
source venv/bin/activate

python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt

```
### Connection to tabmind and transfering files in local network
```
ssh tab@103.225.204.191
ssh sidharth@10.203.1.130
#run script in background
nohup python3 PaddleOCR/tools/train.py -c PaddleOCR/config/rec/arabic_PP-OCRv3_rec.yml &
#checkout nohup.out for logs

#transfer data files
scp -r /var/www/django/data/ tab@103.225.204.191:/var/www/paddle/data/
scp -r /var/www/django/data/ sidharth@10.203.1.130:~/Desktop/django/
scp -r /var/www/django/data/ lohi@10.206.2.223:/var/www/django/data/

#transfer models
scp -r /var/www/django/models/rec/v3/ tab@103.225.204.191:/var/www/paddle/models/rec/
scp -r /var/www/django/models/rec/v2/ lohi@10.206.2.223:/var/www/django/models/rec/v3/training
scp -r sidharth@10.203.1.130:~/Desktop/django/models/rec/v4/training lohi@10.206.4.59:/var/www/django/models/rec/v4_scratch/
scp -r from to
```

### Start training 
- #### Recognition
    Pretrained Model will be found [here](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_train.tar)
    ```
    python3 PaddleOCR/tools/train.py -c PaddleOCR/config/rec/arabic_PP-OCRv3_rec.yml
    ```

    Convert Training to Inference Model
    ```
    python3 ./PaddleOCR/tools/export_model.py -c ./PaddleOCR/config/rec/arabic_PP-OCRv3_rec.yml  -o Global.pretrained_model=./models/rec/v2/training/best_accuracy
    ```

    Inference
    ```
    python3 ./PaddleOCR/tools/infer_rec.py -c ./PaddleOCR/config/rec/arabic_PP-OCRv3_rec.yml -o Global.pretrained_model="./models/rec/v2/training/"
    ```


    ```
    python3 PaddleOCR/tools/train.py -c PaddleOCR/config/rec/arabic_PP-OCRv3_rec.yml -o Global.epoch_num=100
    ```
- #### Detection
    Pretrained Model will be found [here](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar)
    ```
    python3 ./PaddleOCR/tools/train.py -c ./PaddleOCR/config/det/ml_PP-OCRv3_det_cml.yml
    ```

    Convert Training to Inference
    ```
    python3 ./PaddleOCR/tools/export_model.py -c ./PaddleOCR/config/det/ml_PP-OCRv3_det_cml.yml -o Global.pretrained_model=./models/det/v2/training/best_accuracy
    ```

    Inference
    ```
    python3 tools/infer/predict_det.py --image_dir="./test/det/images" --det_model_dir="./inference/det/Multilingual_PP-OCRv3_det_distill_train/v1/"
    ```

- #### Training Visualization
first make visualdir: true in your config file
```
#Recognition
visualdl --logdir "./models/rec/v2/training"

#Detection
visualdl --logdir "./models/det/v2/training"
```

### training script for finetuning
```
python3 PaddleOCR/tools/train.py -c PaddleOCR/config/rec/arabic_PP-OCRv3_rec_finetuning.yml\
 -o\
 Global.save_epoch_step=2\
 Global.save_model_dir=./models/rec/v5/training\
 Global.pretrained_model=./models/rec/v3/training/latest\
 Global.checkpoints=null\
 Global.save_inference_dir=./models/rec/v5/inference\
 Global.character_dict_path=./data/rec/v3/arabic_dict.txt\
 Optimizer.lr.values="[1.0e-04, 2.0e-05]"\
 Optimizer.lr.warmup_epoch=0\
 Train.dataset.data_dir=./media\
 Train.dataset.ext_data_ration_list="[0.1, 1.0]"\
 Train.dataset.label_file_list='["./media/data/synthetic/rec/v1/labels_gt_train.txt","./media/data/original/rec/labels_gt_train.txt"]'\
 Train.loader.batch_size_per_card=112\
 Eval.dataset.data_dir=./media\
 Eval.dataset.label_file_list=./media/data/original/rec/labels_gt_test.txt\
 Eval.loader.batch_size_per_card=112
```

### training script for training from scratch (pretraining)
```
python3 PaddleOCR/tools/train.py -c PaddleOCR/config/rec/arabic_PP-OCRv3_rec_pretraining.yml\
 -o\
 Global.save_epoch_step=2\
 Global.save_model_dir=./models/rec/v5/training\
 Global.pretrained_model=null\
 Global.checkpoints=null\
 Global.save_inference_dir=./models/rec/v5/inference\
 Global.character_dict_path=./data/rec/v3/arabic_dict.txt\
 Optimizer.lr.learning_rate=0.0001\
 Optimizer.lr.warmup_epoch=0\
 Train.dataset.data_dir=./media\
 Train.dataset.ext_data_ration_list="[0.1, 1.0]"\
 Train.dataset.label_file_list='["./media/data/synthetic/rec/v1/labels_gt_train.txt","./media/data/original/rec/labels_gt_train.txt"]'\
 Train.loader.batch_size_per_card=112\
 Eval.dataset.data_dir=./media\
 Eval.dataset.label_file_list=./media/data/original/rec/labels_gt_test.txt\
 Eval.loader.batch_size_per_card=112
```