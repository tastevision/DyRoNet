dataDir="/home/xiang.huang/project/StreamDynamic/data" # path/to/your/data
config="/home/xiang.huang/project/StreamDynamic/cfgs/streamdynamic_DAMO_s.py" # path/to/your/cfg
weights="/home/xiang.huang/project/StreamDynamic/models/checkpoints/streamdynamic_DAMO_branch2_untrained_s.pth" # path/to/your/checkpoint_path
outputDir="/home/xiang.huang/project/StreamDynamic/data/online_result/streamdynamic_DAMO_s" # output dir

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
