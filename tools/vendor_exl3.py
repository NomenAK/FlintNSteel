"""Reproducible import of the EXL3 kernels, without ATen dependency (read-only source)."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get('FNS_EXLLAMAV3_DIR', root / 'external' / 'exllamav3'))
base = source / 'exllamav3/exllamav3_ext'
license = (source / 'LICENSE').read_text(encoding='utf-8')
files = ['ptx.cuh', 'ptx_portable.cuh', 'compat.cuh', 'arch.cuh', 'util.h', 'util.cuh', 'det_gemm.cuh',
         *['quant/' + n for n in ['codebook.cuh', 'exl3_dq.cuh', 'exl3_gemv_kernel.cuh',
          'hadamard_inner.cuh', 'exl3_kernel_map.cuh', 'exl3_gemm_kernel.cuh',
          'exl3_gemm_inner.cuh', 'exl3_devctx.cuh', 'exl3_moe_coop_kernel.cuh', 'exl3_moe_coop.cuh']]]
for name in files:
    text = (base / name).read_text(encoding='utf-8')
    modifications = 'no arithmetic changes'
    if name == 'util.cuh':
        text = text.replace('#include <c10/util/Exception.h>', '#include "check.h"')
        modifications += '; TORCH_CHECK replaced by std::runtime_error'
    if name == 'quant/exl3_moe_coop.cuh':
        text = text.replace('#include <ATen/Tensor.h>\n', '')
        text = text[:text.index('// Static part of the parameter block')]+text[text.index('// Length of the int32 scratch'):]
        modifications += '; Tensor interface removed'
    preamble = '/*\n' + license + '\nOrigin: exllamav3 v1.6.2 (4c842731), Flint patch 3b013de9.\nPath: exllamav3/exllamav3_ext/' + name + '\nModifications: ' + modifications + '.\n*/\n'
    dest = root / 'src/exl3/vendor' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(preamble + text, encoding='utf-8')

# Deterministic router projection and top-k selection: CUDA code only, without the Tensor frontend.
name = 'routing_gemm.cu'
text = (base / name).read_text(encoding='utf-8')
start = text.index('#define RG_')
end = text.index('// Per-device static workspaces')
body = text[start:end]
dest = root / 'src/exl3/vendor/routing_gemm_kernel.cuh'
dest.write_text('/*\n'+license+'\nOrigin: exllamav3 v1.6.2 (4c842731), patch 3b013de9, exllamav3_ext/routing_gemm.cu.\nModifications: Tensor frontend removed, arithmetic unchanged.\n*/\n#pragma once\n#include "det_gemm.cuh"\n'+body, encoding='utf-8')
