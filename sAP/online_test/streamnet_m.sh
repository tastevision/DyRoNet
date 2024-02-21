dataDir="/home/xiang.huang/project/DAMO-StreamNet/data" # path/to/your/data
config="/home/xiang.huang/project/DAMO-StreamNet/cfgs/streamnet_m.py" # path/to/your/cfg
weights="/home/xiang.huang/project/DAMO-StreamNet/models/checkpoints/streamnet_m.pth" # path/to/your/checkpoint_path
outputDir="/home/xiang.huang/project/DAMO-StreamNet/data/online_result/streamnet_m" # output dir

scale=0.5

python online_det.py \
    --data-root "$dataDir/Argoverse-1.1/tracking" \
    --annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
    --fps 30 \
    --weights $weights \
    --in_scale 0.5 \
    --no-mask \
    --out-dir "$outputDir" \
    --overwrite \
    --config $config \

python streaming_eval_scene.py \
    --data-root "$dataDir/Argoverse-1.1/tracking" \
    --annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
    --fps 30 \
    --eta 0 \
    --result-dir "$outputDir" \
    --out-dir "$outputDir" \
    --vis-dir "$outputDir/vis" \
    --vis-scale 0.5 \
    --out-dir "$outputDir" | tee "$outputDir/experiment.log"
    #--overwrite \
