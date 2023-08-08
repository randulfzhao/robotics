"""
手动进行excel命名同步
"""

import os

def sync_excel_to_vid():
    # 获取vid和excel目录下的所有文件
    vid_files = os.listdir('vid')
    excel_files = os.listdir('excel')
    
    for excel_file in excel_files:
        # 去掉扩展名的文件名
        base_name = os.path.splitext(excel_file)[0]
        
        # 检查是否有相对应的视频文件
        matched_video_files = [v for v in vid_files if base_name in v]
        
        # 如果找到了匹配的视频文件
        if matched_video_files:
            # 确保只有一个匹配的视频文件
            if len(matched_video_files) == 1:
                matched_video_file = matched_video_files[0]
                new_excel_name = os.path.splitext(matched_video_file)[0] + '.csv'
                
                # 重命名excel文件
                os.rename(os.path.join('excel', excel_file), os.path.join('excel', new_excel_name))
            else:
                print(f"在'vid'目录中找到了多个与 '{base_name}' 匹配的文件，无法确定要使用哪一个。")
        # 如果在vid文件夹中找不到匹配的文件
        else:
            os.remove(os.path.join('excel', excel_file))
            print(f"文件 '{excel_file}' 已从 'excel' 文件夹中删除。")

if __name__ == "__main__":
    sync_excel_to_vid()