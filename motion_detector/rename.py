import os
import subprocess
import count

def rename_files(original_name, append_string):
    # 定义视频和csv文件的路径
    video_path = os.path.join('vid', original_name)
    csv_path = os.path.join('excel', original_name.split('.')[0] + ".csv")

    # 修改视频名
    if os.path.exists(video_path):
        video_new_name_base = original_name.split('.')[0] + "_edited"
        video_new_name = video_new_name_base + "_" + append_string + "." + original_name.split('.')[1]
        video_new_path = os.path.join('vid', video_new_name)
        os.rename(video_path, video_new_path)

    # 修改csv文件名
    if os.path.exists(csv_path):
        csv_new_name = video_new_name_base + "_" + append_string + ".csv"
        csv_new_path = os.path.join('excel', csv_new_name)
        os.rename(csv_path, csv_new_path)

def delete_file(file_name):
    video_path = os.path.join('vid', file_name)
    csv_path = os.path.join('excel', file_name.split('.')[0] + ".csv")

    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"视频 '{file_name}' 已删除.")

    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"CSV 文件 '{file_name.split('.')[0]}.csv' 已删除.")

def play_video_with_default_player(video_path):
    if os.name == 'nt':  # for Windows
        os.startfile(video_path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"  # for macOS and Linux
        subprocess.call([opener, video_path])

def get_append_string():
    default_values = ["drinkWater", "reachOut", "getPhone"]
    print("请选择一个值进行附加：")
    for i, value in enumerate(default_values):
        print(f"{i+1}. {value}")
    print(f"{len(default_values)+1}. 自定义输入")

    choice = int(input("请输入您的选择（数字）："))
    if 1 <= choice <= len(default_values):
        return default_values[choice-1]
    elif choice == len(default_values) + 1:
        return input("请输入自定义字符串：")
    else:
        print("无效选择。使用默认值 'drinkWater'.")
        return "drinkWater"

def main():
    print("开始执行 main 函数...")
    for video_name in os.listdir('vid'):
        if "_edited" not in video_name:
            video_path = os.path.join('vid', video_name)
            play_video_with_default_player(video_path)

            print(f"视频 {video_name} 预览完毕!")
            option = input("请选择操作：\n1. 重命名文件\n2. 删除文件\n输入选项（1或2）：")

            if option == '1':
                append_string = get_append_string()
                rename_files(video_name, append_string)
            elif option == '2':
                delete_file(video_name)
            else:
                print("无效的选项!")
    print("已完成全部重命名工作，结束 main 函数执行。Cong!")
    count.count_files_in_directory()


if __name__ == "__main__":
    main()