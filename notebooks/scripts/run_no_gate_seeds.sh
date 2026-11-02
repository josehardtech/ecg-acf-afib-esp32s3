#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Variante no_gate (concatenação sem attention gate) com as sementes 43 e 44
# Mesmo protocolo da Tabela 7 e do estudo OAT (partições verificadas contra
# a referência, treino determinístico, mesmo critério de τ).
#
# Uso (com o ambiente pytorch_env ativo):
#   cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/Sensibilidade_OAT/
#   nohup bash run_no_gate_seeds.sh > no_gate_seeds.log 2>&1 &
# Execuções já concluídas (com metrics_final.json) são puladas.
# ═══════════════════════════════════════════════════════════════════════
ROOT=/home/jose_alberto/Modelos_IEEE_Artigo_Revisao
NB=$ROOT/Sensibilidade_OAT/ARCH_variants_v3.ipynb
export OAT_ROOT=$ROOT/Sensibilidade_OAT
export OAT_REF_DIR=$ROOT/Ablation_LOESS_NLM_v3_ok

for SEED in 43 44; do
  NAME=arch_no_gate_seed$SEED
  OUT=$OAT_ROOT/$NAME
  if [ -f "$OUT/metrics_final.json" ]; then
    echo "=== $NAME já concluído, pulando ==="; continue
  fi
  mkdir -p "$OUT"
  echo "=== Iniciando $NAME (arch=no_gate, seed=$SEED): $(date) ==="
  OAT_ARCH=no_gate OAT_PARAM=arch OAT_VALUE= OAT_NAME=$NAME OAT_SEED=$SEED \
    jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name=pytorch_env \
      --output-dir "$OUT" --output "ARCH_${NAME}.ipynb" "$NB"
  RC=$?
  echo "=== Concluído $NAME: $(date) | código de saída $RC ==="
done
echo "=== Sementes no_gate finalizadas: $(date) ==="
