#!/bin/bash
# two branches
. ./script/DAMO_s_DAMO_m_lora_eval.sh
. ./script/DAMO_s_DAMO_l_lora_eval.sh
. ./script/DAMO_m_DAMO_l_lora_eval.sh
. ./script/LSN_s_LSN_m_lora_eval.sh
. ./script/LSN_s_LSN_l_lora_eval.sh
. ./script/LSN_m_LSN_l_lora_eval.sh
. ./script/SYOLO_s_SYOLO_m_lora_eval.sh
. ./script/SYOLO_s_SYOLO_l_lora_eval.sh
. ./script/SYOLO_m_SYOLO_l_lora_eval.sh

# two branches (different model)
. ./script/DAMO_s_LSN_s_lora_eval.sh
. ./script/DAMO_s_LSN_m_lora_eval.sh
. ./script/DAMO_s_LSN_l_lora_eval.sh
. ./script/DAMO_m_LSN_s_lora_eval.sh
. ./script/DAMO_l_LSN_s_lora_eval.sh

# three branches
. ./script/DAMO_s_DAMO_m_DAMO_l_lora_eval.sh
. ./script/LSN_s_LSN_m_LSN_l_lora_eval.sh
. ./script/SYOLO_s_SYOLO_m_SYOLO_l_lora_eval.sh
