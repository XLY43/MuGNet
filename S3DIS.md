# S3DIS

Download [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html) and extract `Stanford3dDataset_v1.2_Aligned_Version.zip` to `$S3DIS_DIR/data`, where `$S3DIS_DIR` is set to dataset directory.

Define $S3DIS_DIR as the location of the folder containing `/data`

## Learned Partition

To learn the partition from scratch run:
```
python supervized_partition/graph_processing.py --ROOT_PATH $S3DIS_DIR --dataset s3dis --voxel_width 0.03; \

for FOLD in 1 2 3 4 5 6; do \
    python ./supervized_partition/supervized_partition.py --ROOT_PATH $S3DIS_DIR  --cvfold $FOLD \
    --odir results_partition/s3dis/best --epochs 50 --reg_strength 0.1 --spatial_emb 0.2  \
    --global_feat eXYrgb --CP_cutoff 25; \
done
```
Or download the trained weights [here](http://recherche.ign.fr/llandrieu/SPG/S3DIS/pretrained.zip) in the folder `results_partition/s3dis/pretrained`, unzipped and run the following code:

```
for FOLD in 1 2 3 4 5 6; do \
    python ./supervized_partition/supervized_partition.py --ROOT_PATH $S3DIS_DIR  --cvfold $FOLD --epochs -1 \
    --odir results_partition/s3dis/pretrained --reg_strength 0.1 --spatial_emb 0.2 --global_feat eXYrgb \
    --CP_cutoff 25 --resume RESUME; \
done
```

To evaluate the quality of the partition, run:
```
python supervized_partition/evaluate_partition.py --dataset s3dis --folder pretrained --cvfold 123456
```

Then, reorganize point clouds into superpoints with:
```
python learning/s3dis_dataset.py --S3DIS_PATH $S3DIS_DIR --supervized_partition 1  -plane_model_elevation 1
```

## MsGNet Training and Testing
Then to learn the MsGNet models from scratch, run:
```
for FOLD in 1 2 3 4 5 6; do \
	CUDA_VISIBLE_DEVICES=0 python ./learning/main.py --dataset s3dis --S3DIS_PATH $S3DIS_DIR --batch_size 5 \
  --cvfold $FOLD --epochs 250 --lr_steps '[150,200]' --model_config "gru_10_0,f_13" --ptn_nfeat_stn 10 \
  --nworkers 2 --spg_augm_order 5 --pc_attribs xyzXYZrgbe --spg_augm_hardcutoff 768 --ptn_minpts 50 \
  --use_val_set 1 --odir results/s3dis/best/cv$FOLD; \
    done;
```

Or use [trained weights](https://cmu.box.com/s/41jrvat2mpx6vt5qrj5qng3gwk8z8rfc) with `--epochs -1` and `--resume RESUME`:
```
for FOLD in 1 2 3 4 5 6; do \
	CUDA_VISIBLE_DEVICES=1 python ./learning/main.py --dataset s3dis --S3DIS_PATH $S3DIS_DIR --batch_size 5 \
  --cvfold $FOLD --epochs -1 --lr_steps '[150,200]' --model_config "gru_10_0,f_13" --ptn_nfeat_stn 10 \
  --nworkers 2 --spg_augm_order 5 --pc_attribs xyzXYZrgbe --spg_augm_hardcutoff 768 --ptn_minpts 50 \
  --use_val_set 1 --odir results/s3dis/pretrained/cv$FOLD --resume RESUME; \
    done;
```
Note that these weights are specifically adapted to the pretrained model for the learned partition. Any change to the partition might decrease their performance. 
```
To evaluate the 6-fold performance:
```
python learning/evaluate.py --dataset s3dis --odir results/s3dis/best --cvfold 123456
```
To visualize the segmentation results, edit the files you want in visualize.py and run the following:
```
python partition/visualize.py --dataset s3dis --ROOT_PATH $S3DIS_DIR --res_file results/s3dis/pretrained/cv1/predictions_test --file_path Area_1/conferenceRoom_1 --output_type igfpres
```

