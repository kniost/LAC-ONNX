"""LAC-ONNX 使用示例"""

from lac_onnx import LAC

lac = LAC()

# 单句分析
print('=== 单句 ===')
result = lac.run('张三在北京市工作，手机号13812345678')
for word, tag in result:
    print(f'  {word:12s} {tag}')

# 批量分析
print('\n=== 批量 ===')
texts = [
    '李四和欧阳明月去了上海交通大学',
    '中国人民银行发布了2024年第一季度货币政策报告',
]
results = lac.run(texts)
for text, result in zip(texts, results):
    print(f'\n>>> {text}')
    for word, tag in result:
        print(f'  {word:12s} {tag}')

# NER 过滤
print('\n=== 仅 NER 实体 ===')
text = '张三和李四在北京市朝阳区的中国银行办理业务'
for word, tag in lac.run(text):
    if tag in ('PER', 'LOC', 'ORG'):
        print(f'  {word:12s} {tag}')
