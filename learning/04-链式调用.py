class Chain:
    def __init__(self, value):
        self.value = value

    def __or__(self, other):
        # 调用 | 运算符 触发的方法
        return other(self.value)

def prompt(text):
    return "请求回答问题:{}".format(text)

aa = Chain('人工智能是什么?')

res = aa | prompt
print(res)