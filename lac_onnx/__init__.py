"""LAC-ONNX — 百度 LAC 中文词法分析模型的 ONNX 版本

仅依赖 onnxruntime + numpy，无需 PaddlePaddle。

用法::

    from lac_onnx import LAC

    lac = LAC()
    result = lac.run('张三在北京市工作')
    # [('张三', 'PER'), ('在', 'p'), ('北京市', 'LOC'), ('工作', 'n')]
"""

__version__ = '0.1.0'

from lac_onnx.lac import LAC

__all__ = ['LAC']
