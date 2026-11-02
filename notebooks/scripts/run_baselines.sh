#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Baselines arquiteturais E5–E7 — mesmo split por paciente da referência.
#   E5  MobileNet-1D (INT8)          -> arch_mobilenet/
#   E6  CNN rasa + LSTM (float32)    -> base_lstm/
#   E7  Random forest e SVM com features R-R (float32) -> base_rf_rr/, base_svm_rr/
# Pasta de trabalho: $ROOT/Baselines_E5_E7 (códigos e resultados)
# Uso (com pytorch_env ativo):  nohup bash run_baselines.sh > baselines_execucao.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════
ROOT=/home/jose_alberto/Modelos_IEEE_Artigo_Revisao
DIR=$ROOT/Baselines_E5_E7
export OAT_ROOT=$DIR
export OAT_REF_DIR=$ROOT/Ablation_LOESS_NLM_v3_ok
run_nb () {   # nome  notebook  arquitetura  arquivo-de-conclusão
  NAME=$1; NB=$2; ARCH=$3; DONE=$4
  if [ -f "$DONE" ]; then echo "=== $NAME já concluído, pulando ==="; return; fi
  mkdir -p "$DIR/$NAME"
  echo "=== Iniciando $NAME: $(date) ==="
  OAT_NAME=$NAME OAT_PARAM=arch OAT_VALUE=$ARCH OAT_SEED=42 OAT_ARCH=$ARCH \
    jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=pytorch_env \
      --output-dir "$DIR/$NAME" --output "RUN_${NAME}.ipynb" "$DIR/$NB"
  echo "=== Concluído $NAME: $(date) | código de saída $? ==="
}
run_nb arch_mobilenet ARCH_variants_v3.ipynb mobilenet "$DIR/arch_mobilenet/metrics_final.json"
run_nb base_lstm      BASELINE_lstm_v3.ipynb  lstm      "$DIR/base_lstm/metrics_final.json"
run_nb base_rr        BASELINE_rr_v3.ipynb    dual      "$DIR/base_svm_rr/metrics_final.json"
echo "=== Baselines finalizados: $(date) ==="
