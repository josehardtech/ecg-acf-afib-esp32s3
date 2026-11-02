#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Estudo OAT (one-at-a-time) de sensibilidade a hiperparâmetros
# Ablation #1 v3 — executa todas as variações em sequência.
# Uso (com o ambiente pytorch_env ativo):
#   nohup bash run_oat.sh > oat_execucao.log 2>&1 &
# Variações já concluídas (com metrics_final.json) são puladas, então o
# script pode ser reiniciado sem refazer o que já terminou.
# ═══════════════════════════════════════════════════════════════════════
ROOT=/home/jose_alberto/Modelos_IEEE_Artigo_Revisao
NB=$ROOT/Sensibilidade_OAT/OAT_sensitivity_v3.ipynb
export OAT_ROOT=$ROOT/Sensibilidade_OAT
export OAT_REF_DIR=$ROOT/Ablation_LOESS_NLM_v3_ok

# nome | parâmetro | valor | semente de treino
RUNS=(
  "reference_seed42|reference||42"
  "gamma_1.0|gamma|1.0|42"
  "gamma_3.0|gamma|3.0|42"
  "alpha_0.5|alpha|0.5|42"
  "alpha_0.9|alpha|0.9|42"
  "dropout_0.1-0.1|dropout|0.1/0.1|42"
  "dropout_0.5-0.4|dropout|0.5/0.4|42"
  "lr_2e-3|lr|2e-3|42"
  "lr_8e-3|lr|8e-3|42"
  "wd_1e-5|wd|1e-5|42"
  "wd_1e-3|wd|1e-3|42"
  "reference_seed43|seed||43"
  "reference_seed44|seed||44"
)

for r in "${RUNS[@]}"; do
  IFS='|' read -r NAME PARAM VALUE SEED <<< "$r"
  OUT=$OAT_ROOT/$NAME
  if [ -f "$OUT/metrics_final.json" ]; then
    echo "=== $NAME já concluído, pulando ==="; continue
  fi
  mkdir -p "$OUT"
  echo "=== Iniciando $NAME ($PARAM=$VALUE, seed=$SEED): $(date) ==="
  OAT_NAME=$NAME OAT_PARAM=$PARAM OAT_VALUE=$VALUE OAT_SEED=$SEED \
    jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name=pytorch_env \
      --output-dir "$OUT" --output "OAT_${NAME}.ipynb" "$NB"
  RC=$?
  echo "=== Concluído $NAME: $(date) | código de saída $RC ==="
done
echo "=== Estudo OAT finalizado: $(date) ==="
