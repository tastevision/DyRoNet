#!/bin/bash
# two branches (same model)
. ./script/DAMO_s_DAMO_m.sh
. ./script/DAMO_s_DAMO_l.sh
. ./script/DAMO_m_DAMO_l.sh
. ./script/LSN_s_LSN_m.sh
. ./script/LSN_s_LSN_l.sh
. ./script/LSN_m_LSN_l.sh
. ./script/SYOLO_s_SYOLO_m.sh
. ./script/SYOLO_s_SYOLO_l.sh
. ./script/SYOLO_m_SYOLO_l.sh

# two branches (different model)
. ./script/DAMO_s_LSN_s.sh
. ./script/DAMO_s_LSN_m.sh
. ./script/DAMO_s_LSN_l.sh
. ./script/DAMO_m_LSN_s.sh
. ./script/DAMO_l_LSN_s.sh

# three branches
. ./script/DAMO_s_DAMO_m_DAMO_l.sh
. ./script/LSN_s_LSN_m_LSN_l.sh
. ./script/SYOLO_s_SYOLO_m_SYOLO_l.sh
