#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Arquiteturas alternativas (E1–E4) — mesmo protocolo do estudo OAT.
# Uso (com pytorch_env ativo):  nohup bash run_arch.sh > arch_execucao.log 2>&1 &
# Variações já concluídas (com metrics_final.json) são puladas.
# ═══════════════════════════════════════════════════════════════════════
ROOT=/home/jose_alberto/Modelos_IEEE_Artigo_Revisao
NB=$ROOT/Sensibilidade_OAT/ARCH_variants_v3.ipynb
export OAT_ROOT=$ROOT/Sensibilidade_OAT
export OAT_REF_DIR=$ROOT/Ablation_LOESS_NLM_v3_ok

# nome | arquitetura
RUNS=(
  "arch_cnn_only|cnn_only"
  "arch_no_gate|no_gate"
  "arch_early_fusion|early_fusion"
  "arch_acf_trunc|acf_trunc"
)
for r in "${RUNS[@]}"; do
  IFS='|' read -r NAME ARCH <<< "$r"
  OUT=$OAT_ROOT/$NAME
  if [ -f "$OUT/metrics_final.json" ]; then echo "=== $NAME já concluído, pulando ==="; continue; fi
  mkdir -p "$OUT"
  echo "=== Iniciando $NAME (arch=$ARCH): $(date) ==="
  OAT_NAME=$NAME OAT_PARAM=arch OAT_VALUE=$ARCH OAT_SEED=42 OAT_ARCH=$ARCH \
    jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name=pytorch_env \
      --output-dir "$OUT" --output "ARCH_${NAME}.ipynb" "$NB"
  echo "=== Concluído $NAME: $(date) | código de saída $? ==="
done
echo "=== Arquiteturas alternativas finalizadas: $(date) ==="
