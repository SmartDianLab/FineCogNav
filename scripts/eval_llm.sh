# cd ./FineCogNav
echo $PWD

# export DASHSCOPE_API_KEY=""
# python -u ./airsim_plugin/AirVLNSimulatorServerTool.py --port 30001 --gpus 2

# bash ./scripts/eval_llm.sh qwen3.5-397b-a17b qwen3.5-397b-a17b rl_4 30001
# Qwen/Qwen2.5-72B-Instruct
# Qwen/Qwen2.5-VL-32B-Instruct
# val_unseen
# 30001

python -u ./src/vlnce_src/eval_llm.py \
--EVAL_GENERATE_VIDEO \
--SAVE_LOG \
--name LLM \
--batchSize 1 \
--EVAL_LLM $1 \
--EVAL_VLM $2 \
--EVAL_DATASET $3 \
--EVAL_NUM -1 \
--Image_Width_RGB 672 \
--Image_Height_RGB 672 \
--Image_Width_DEPTH 672 \
--Image_Height_DEPTH 672 \
--maxAction 200 \
--simulator_tool_port $4