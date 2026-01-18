setlocal EnableDelayedExpansion

set PYTHON=py
set SCRIPT=merge.py

REM ========================
REM BASELINE
REM ========================
set BASE_BS=50
set BASE_LR=5e-3
set BASE_TKZ=AdamW
set BASE_EMB=100
set BASE_KS=3,4
set BASE_CF=50
set BASE_DO=0.5

REM ========================
REM HYPERPARAMETER ABLATION
REM ========================
set LIST_BS=25 100
set LIST_LR=5e-2 5e-4
set LIST_TKZ=Muon
set LIST_EMB=50 200
set LIST_KS=3;3 4 5
set LIST_CF=25 100
set LIST_DO=0.4 0.6

for %%B in (%LIST_BS%) do (
  %PYTHON% %SCRIPT% ^
    --use_wandb ^
    --batch_size %%B ^
    --lr %BASE_LR% ^
    --tokenizer %BASE_TKZ% ^
    --embed_dim %BASE_EMB% ^
    --kernel_size %BASE_KS% ^
    --conv_filters %BASE_CF% ^
    --dropout %BASE_DO% ^
    --wandb_group "LightTextCNN batch_size ablation" ^
    --wandb_note "LightTextCNN batch_size ablation"
)

for %%L in (%LIST_LR%) do (
  %PYTHON% %SCRIPT% ^
    --use_wandb ^
    --batch_size %BASE_BS% ^
    --lr %%L ^
    --tokenizer %BASE_TKZ% ^
    --embed_dim %BASE_EMB% ^
    --kernel_size %BASE_KS% ^
    --conv_filters %BASE_CF% ^
    --dropout %BASE_DO% ^
    --wandb_group "LightTextCNN learning_rate ablation" ^
    --wandb_note "LightTextCNN learning_rate ablation"
)

for %%T in (%LIST_TKZ%) do (
  %PYTHON% %SCRIPT% ^
    --use_wandb ^
    --batch_size %BASE_BS% ^
    --lr %BASE_LR% ^
    --tokenizer %%T ^
    --embed_dim %BASE_EMB% ^
    --kernel_size %BASE_KS% ^
    --conv_filters %BASE_CF% ^
    --dropout %BASE_DO% ^
    --wandb_group "LightTextCNN tokenizer ablation" ^
    --wandb_note "LightTextCNN tokenizer ablation"
)

for %%E in (%LIST_EMB%) do (
  %PYTHON% %SCRIPT% ^
    --use_wandb ^
    --batch_size %BASE_BS% ^
    --lr %BASE_LR% ^
    --tokenizer %BASE_TKZ% ^
    --embed_dim %%E ^
    --kernel_size %BASE_KS% ^
    --conv_filters %BASE_CF% ^
    --dropout %BASE_DO% ^
    --wandb_group "LightTextCNN embedding_dimention ablation" ^
    --wandb_note "LightTextCNN embedding_dimention ablation"
)

for %%K in (%LIST_KS%) do (
  set KS_STR=%%K
  set KS_STR=!KS_STR:;= !

  %PYTHON% %SCRIPT% ^
    --use_wandb ^
    --batch_size %BASE_BS% ^
    --lr %BASE_LR% ^
    --tokenizer %BASE_TKZ% ^
    --embed_dim %BASE_EMB% ^
    --kernel_size !KS_STR! ^
    --conv_filters %BASE_CF% ^
    --dropout %BASE_DO% ^
    --wandb_group "LightTextCNN kernel_size ablation" ^
    --wandb_note "LightTextCNN kernel_size ablation"
)

for %%C in (%LIST_CF%) do (
  %PYTHON% %SCRIPT% ^
    --use_wandb ^
    --batch_size %BASE_BS% ^
    --lr %BASE_LR% ^
    --tokenizer %BASE_TKZ% ^
    --embed_dim %BASE_EMB% ^
    --kernel_size %BASE_KS% ^
    --conv_filters %%C ^
    --dropout %BASE_DO% ^
    --wandb_group "LightTextCNN convolution_filter ablation" ^
    --wandb_note "LightTextCNN convolution_filter ablation"
)

for %%D in (%LIST_DO%) do (
  %PYTHON% %SCRIPT% ^
    --use_wandb ^
    --batch_size %BASE_BS% ^
    --lr %BASE_LR% ^
    --tokenizer %BASE_TKZ% ^
    --embed_dim %BASE_EMB% ^
    --kernel_size %BASE_KS% ^
    --conv_filters %BASE_CF% ^
    --dropout %%D ^
    --wandb_group "LightTextCNN dropout ablation" ^
    --wandb_note "LightTextCNN dropout ablation"
)