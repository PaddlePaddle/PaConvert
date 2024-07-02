import sys
sys.path.append('tests/code_library/code_case/convert_paddle_code/utils')
import paddle_aux
import paddlenlp
import paddle
print('#########################case1#########################')


class QWenPreTrainedModel(paddlenlp.transformers.PretrainedModel):
    pass


print('#########################case2#########################')


class QWenTokenizer(paddlenlp.transformers.PretrainedTokenizer):
    pass


print('#########################case3#########################')
paddle_aux.apply_rotary_emb_func(x=x, cos=cos, sin=sin)
print('#########################case4#########################')
paddle_aux.rms_norm(x=x, weight=weight, epsilon=eps)
