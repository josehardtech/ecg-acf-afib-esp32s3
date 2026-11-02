#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Duas execuções do ARCH_variants_v4.ipynb (arquitetura de referência, semente 42):
#   v4_reference_seed42    escalonador global (igual ao v3) — deve reproduzir o modelo
#                          implantado bit a bit e salva as probabilidades float32
#   v4_fold_scaler_seed42  escalonador da ACF ajustado dentro de cada fold
#
# Uso (com o ambiente pytorch_env ativo):
#   cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Sensibilidade_OAT/
#   nohup bash run_v4.sh > v4.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════
ROOT=/home/jose_alberto/Modelos_IEEE_Artigo_Revisao
NB=$ROOT/Sensibilidade_OAT/ARCH_variants_v4.ipynb
export OAT_ROOT=$ROOT/Sensibilidade_OAT
export OAT_REF_DIR=$ROOT/Ablation_LOESS_NLM_v3_ok

for RUN in "v4_reference_seed42|0" "v4_fold_scaler_seed42|1"; do
  IFS='|' read -r NAME FS <<< "$RUN"
  OUT=$OAT_ROOT/$NAME
  if [ -f "$OUT/metrics_final.json" ]; then
    echo "=== $NAME já concluído, pulando ==="; continue
  fi
  mkdir -p "$OUT"
  echo "=== Iniciando $NAME (fold_scaler=$FS): $(date) ==="
  OAT_ARCH=dual OAT_PARAM=arch OAT_VALUE= OAT_NAME=$NAME OAT_SEED=42 OAT_FOLD_SCALER=$FS \
    jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name=pytorch_env \
      --output-dir "$OUT" --output "ARCH_${NAME}.ipynb" "$NB"
  RC=$?
  echo "=== Concluído $NAME: $(date) | código de saída $RC ==="
done
echo "=== Execuções v4 finalizadas: $(date) ==="
