"""
Gera os arquivos do firmware para QUALQUER modelo INT8 dos notebooks v3/v4
(modelo implantado, MobileNet-1D, etc.), detectando a partir do próprio .tflite:
  - quais entradas existem (input_raw, input_acf) e se a ACF é de fato usada;
  - a lista exata de operações para o MicroMutableOpResolver;
  - quantização, τ do firmware e limiar inteiro.

Saídas (em --out):
  model.h          bytes do .tflite
  model_params.h   SHA-256, τ, limiar inteiro, quantização, índices e uso da ACF
  model_ops.inc    chamadas resolver->AddXxx() para as operações do modelo
  scaler_params.h  StandardScaler da ACF (vazio se o modelo não usa ACF)

Uso:
  cd /home/jose_alberto/Modelos_IEEE_Artigo_Revisao/
  # modelo implantado
  python export_firmware.py --model-dir Ablation_LOESS_NLM_v3_ok --out firmware_v3 --tag "v3 No-LOESS/NLM"
  # MobileNet-1D
  python export_firmware.py --model-dir Sensibilidade_OAT/arch_mobilenet --out firmware_mobilenet --tag "MobileNet-1D baseline"
"""
import argparse
import glob
import hashlib
import os
import pickle
from datetime import datetime

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    OpResolverType = tf.lite.experimental.OpResolverType

# Nome da operação no LiteRT -> método do MicroMutableOpResolver (esp-tflite-micro 1.3.x)
OP_MAP = {
    'ADD': 'AddAdd', 'AVERAGE_POOL_2D': 'AddAveragePool2D', 'CONCATENATION': 'AddConcatenation',
    'CONV_2D': 'AddConv2D', 'DEPTHWISE_CONV_2D': 'AddDepthwiseConv2D', 'DEQUANTIZE': 'AddDequantize',
    'EXPAND_DIMS': 'AddExpandDims', 'FULLY_CONNECTED': 'AddFullyConnected', 'HARD_SWISH': 'AddHardSwish',
    'LOGISTIC': 'AddLogistic', 'MAX_POOL_2D': 'AddMaxPool2D', 'MEAN': 'AddMean', 'MUL': 'AddMul',
    'PACK': 'AddPack', 'PAD': 'AddPad', 'PADV2': 'AddPadV2', 'QUANTIZE': 'AddQuantize', 'RELU': 'AddRelu',
    'RELU6': 'AddRelu6', 'RESHAPE': 'AddReshape', 'SHAPE': 'AddShape', 'SOFTMAX': 'AddSoftmax',
    'SQUEEZE': 'AddSqueeze', 'STRIDED_SLICE': 'AddStridedSlice', 'SUB': 'AddSub', 'SUM': 'AddSum',
    'TRANSPOSE': 'AddTranspose', 'UNPACK': 'AddUnpack', 'SPLIT': 'AddSplit', 'TANH': 'AddTanh',
}
SYMBOL = 'ecg_hybrid_v15_sem_loess_nlm_int8_tflite'   # mesmo símbolo do firmware atual

ap = argparse.ArgumentParser()
ap.add_argument('--model-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--tag', default='model')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

tfl = sorted(glob.glob(os.path.join(a.model_dir, '*int8*.tflite')))
assert len(tfl) == 1, f'esperado 1 .tflite INT8 em {a.model_dir}, achei {tfl}'
TFLITE = tfl[0]
thr_f = sorted(glob.glob(os.path.join(a.model_dir, '*threshold_tflite*.pkl')))
assert thr_f, f'pkl do τ INT8 não encontrado em {a.model_dir}'
tau = float(pickle.load(open(thr_f[0], 'rb'))['tau_firmware'])
blob = open(TFLITE, 'rb').read()
sha = hashlib.sha256(blob).hexdigest()

# Sem delegates (XNNPACK), para que a lista de operações seja a do grafo original
it = Interpreter(model_path=TFLITE,
                 experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
it.allocate_tensors()
ins, outs = it.get_input_details(), it.get_output_details()
i_raw = next((i for i, d in enumerate(ins) if 'raw' in d['name'].lower()), None)
i_acf = next((i for i, d in enumerate(ins) if 'acf' in d['name'].lower()), None)
assert i_raw is not None, 'modelo sem entrada input_raw'
ops = it._get_ops_details()
op_names = sorted({o['op_name'] for o in ops})
assert not any('DELEGATE' in o.upper() for o in op_names), f'operações delegadas: {op_names}' 
acf_used = False
if i_acf is not None:
    t_acf = ins[i_acf]['index']
    acf_used = any(t_acf in list(o['inputs']) for o in ops)
missing = [o for o in op_names if o not in OP_MAP]
assert not missing, f'operações sem mapeamento para o resolver: {missing} — acrescentar em OP_MAP'

rs, rz = ins[i_raw]['quantization']
as_, az = ins[i_acf]['quantization'] if i_acf is not None else (1.0, 0)
os_, oz = outs[0]['quantization']
q_thr = int(round(tau / os_ + oz))
stamp = datetime.now().isoformat(timespec='seconds')
print(f'Modelo: {TFLITE}\n  SHA-256 {sha}\n  {len(blob)} bytes | entradas: {len(ins)} '
      f'(raw idx {i_raw}{"" if i_acf is None else f", acf idx {i_acf}"}) | ACF usada: {acf_used}')
print(f'  τ = {tau}  limiar inteiro = {q_thr}  | raw scale={rs:.8f} zp={rz}'
      + ('' if i_acf is None else f' | acf scale={as_:.8f} zp={az}') + f' | out scale={os_:.8f} zp={oz}')
print(f'  operações ({len(op_names)}): {", ".join(op_names)}')

with open(f'{a.out}/model.h', 'w') as f:
    f.write(f'/* model.h — {a.tag} — gerado por export_firmware.py em {stamp}\n'
            f' * SHA-256 {sha} ({len(blob)} bytes). Incluir em UM único arquivo .cc. */\n#pragma once\n\n'
            f'alignas(16) const unsigned char {SYMBOL}[] = {{\n')
    for k in range(0, len(blob), 16):
        f.write('    ' + ', '.join(f'0x{b:02x}' for b in blob[k:k + 16]) + ',\n')
    f.write(f'}};\nconst unsigned int {SYMBOL}_len = {len(blob)};\n')

with open(f'{a.out}/model_params.h', 'w') as f:
    f.write(f'''/* model_params.h — {a.tag} — gerado por export_firmware.py em {stamp} */
#pragma once
#define MODEL_SHA256          "{sha}"
#define MODEL_VERSION_TAG     "{a.tag}"
#define MODEL_TFLITE_BYTES    {len(blob)}
#define MODEL_NUM_INPUTS      {len(ins)}
#define MODEL_USES_ACF        {1 if acf_used else 0}
#define MODEL_NUM_OPS         {len(op_names)}
/* Decisão: AFIB se saída INT8 >= MODEL_THRESHOLD_Q  (equivale a p >= MODEL_TAU) */
#define MODEL_TAU             {tau:.8f}f
#define MODEL_THRESHOLD_Q     ({q_thr})
#define MODEL_RAW_INPUT_INDEX {i_raw}
#define MODEL_ACF_INPUT_INDEX {-1 if i_acf is None else i_acf}
#define MODEL_RAW_SCALE       {rs:.10e}f
#define MODEL_RAW_ZP          ({int(rz)})
#define MODEL_ACF_SCALE       {as_:.10e}f
#define MODEL_ACF_ZP          ({int(az)})
#define MODEL_OUT_SCALE       {os_:.10e}f
#define MODEL_OUT_ZP          ({int(oz)})
''')

with open(f'{a.out}/model_ops.inc', 'w') as f:
    f.write(f'/* model_ops.inc — {a.tag} — operações do modelo ({len(op_names)}) */\n')
    for o in op_names:
        f.write(f'    resolver->{OP_MAP[o]}();   /* {o} */\n')

with open(f'{a.out}/scaler_params.h', 'w') as f:
    f.write(f'/* scaler_params.h — {a.tag} — gerado por export_firmware.py em {stamp}\n'
            ' * Requer TFLITE_ACF_LAGS (tflite_inference.h). */\n#pragma once\n\n')
    if acf_used:
        sc_f = sorted(glob.glob(os.path.join(a.model_dir, '*scaler*.pkl')))
        assert sc_f, f'escalonador não encontrado em {a.model_dir}'
        sc = pickle.load(open(sc_f[0], 'rb'))
        assert sc.mean_.shape == (5000,), f'escalonador com {sc.mean_.shape} lags (firmware espera 5.000)'
        for name, v in (('kScalerMean', sc.mean_), ('kScalerScale', sc.scale_)):
            lines = [', '.join(f'{x:.8e}f' for x in v[k:k + 6]) for k in range(0, len(v), 6)]
            f.write(f'static const float {name}[TFLITE_ACF_LAGS] = {{\n    ' + ',\n    '.join(lines) + '\n};\n\n')
    else:
        f.write('/* Este modelo não usa a ACF: escalonador não necessário. */\n'
                'static const float kScalerMean[1]  = {0.0f};\nstatic const float kScalerScale[1] = {1.0f};\n')
print(f'Gerados em {a.out}/: model.h | model_params.h | model_ops.inc | scaler_params.h')
