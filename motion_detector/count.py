import os

def count_files_in_directory(dir_path='vid'):
    # 定义文件结尾的种类
    suffixes = ['drinkWater', 'reachOut', 'getPhone']
    
    # 初始化计数器
    counts = {suffix: 0 for suffix in suffixes}
    total_files = 0

    # 遍历文件夹内容
    for file_name in os.listdir(dir_path):
        for suffix in suffixes:
            if suffix in file_name:
                counts[suffix] += 1
                total_files += 1
                break  # 如果匹配到后缀，则不再继续检查其他后缀，直接处理下一个文件

    # 打印结果
    for suffix, count in counts.items():
        print(f"Number of samples for motion '{suffix}' is {count}")
    print(f"Total files: {total_files}")

# 调用函数
count_files_in_directory()
