from functools import wraps,partial
import logging

def attach_wrapper(obj, func=None):
    # 这个函数用于将装饰器附加到对象上。
    # 如果没有指定函数，则返回一个函数，该函数可以将装饰器附加到任何其他函数上。
    # 如果指定了函数，则将装饰器附加到该函数上并返回该函数。

    if func is None:
        return partial(attach_wrapper, obj)
    setattr(obj, func.__name__, func)
    return func

def logged(level, name=None, message=None):
    # 这个函数用于创建一个装饰器，该装饰器可以为函数添加日志记录功能。
    # 参数 `level` 指定日志记录级别。
    # 参数 `name` 指定日志记录器的名称。如果没有指定，则使用函数所在的模块名称。
    # 参数 `message` 指定日志记录消息。如果没有指定，则使用函数名称。

    def decorate(func):
        # 这个函数用于将装饰器附加到函数上。

        logname = name if name else func.__module__
        log = logging.getLogger(logname)
        logmsg = message if message else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 这个函数是被装饰函数的包装函数。它会在调用被装饰函数之前记录一条日志。

            log.log(level, logmsg)
            return func(*args, **kwargs)

        @attach_wrapper(wrapper)
        def set_level(newlevel):
            # 这个函数用于设置日志记录级别。

            nonlocal level
            level = newlevel

        @attach_wrapper(wrapper)
        def set_message(newmsg):
            # 这个函数用于设置日志记录消息。

            nonlocal logmsg
            logmsg = newmsg

        return wrapper

    return decorate

# Example use
@logged(logging.DEBUG)
def add(x,y):
    return x+y

@logged(logging.CRITICAL,'example')
def spam():
    print('Spam!')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print(add(2,3))
    add.set_message('Add called')
    print(add(2,3))
    add.set_level(logging.WARNING)
    print(add(2,3))
    spam()
    spam.set_message('Spam called')
    spam()
    spam.set_level(logging.WARNING)
    spam()

    # def Data_Padding_As_Beach_name(self, data, Features):
    #     data = data.copy()
    #     columns_to_process = Features
        
    #     # 填充负数值和缺失值
    #     for column in columns_to_process:
    #         # Fill missing and negative values with the mean of non-negative values for each beach
    #         filled_values = data.groupby('Beach_Name')[column].apply(lambda x: x.mask(x < 0, x[x >= 0].mean()).fillna(x[x >= 0].mean())).reset_index(level=0, drop=True)
    #         data[column] = filled_values
        

    #     data_padded = pd.DataFrame(data)
    #     return data_padded