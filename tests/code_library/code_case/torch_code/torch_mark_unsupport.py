from torch.utils.cpp_extension import load_inline
print("#########################case1#########################")
source = """
at::Tensor sin_add(at::Tensor x, at::Tensor y) {
    return x.sin() + y.sin();
}
"""
load_inline(
        name='inline_extension',
        cpp_sources=[source],
        functions=['sin_add'])
result = True
print("#########################case2#########################")
print("(Torch")
print("#########################case3#########################")
""" This is an non-standard example
) """
""" (
    This is an non-standard example
) """
""" (
    This is (an non-standard example
) """
print("#########################case4#########################")
# TODO: Uncomment after merging its conversion strategy
# code_convert_tools_from_torch_to_paddle_logits_processor = LogitsProcessorList([stop_words_logits_processor])
print("#########################case5#########################")
""" This is an non-standard example) """
source = """
at::Tensor sin_add(at::Tensor x, at::Tensor y) {
    return x.sin() + y.sin();
}
"""
load_inline(
        name='inline_extension',
        cpp_sources=[source],
        functions=['sin_add'])
result = True
