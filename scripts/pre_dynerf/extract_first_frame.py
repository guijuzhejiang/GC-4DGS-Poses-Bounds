import os
import cv2
import re
import sys

def extract_first_frame(input_directory):
    """
    从指定目录的MP4视频文件中提取第一帧图像，并保存到image_colmap子目录中。

    @param input_directory: 存放MP4视频文件的输入目录路径。
    @type input_directory: str
    """
    output_image_directory = os.path.join(input_directory, "image_colmap")
    os.makedirs(output_image_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_directory, filename)
            
            # 使用正则表达式从文件名中提取数字
            match = re.search(r'cam(\d+)\.mp4', filename)
            if match:
                video_number = int(match.group(1))
                output_image_name = f"r_{video_number:03d}.png"
                output_image_path = os.path.join(output_image_directory, output_image_name)

                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"Error: Could not open video file {filename}")
                    continue

                ret, frame = cap.read()

                if ret:
                    cv2.imwrite(output_image_path, frame)
                    print(f"Extracted first frame from {filename} and saved as {output_image_name}")
                else:
                    print(f"Error: Could not read first frame from {filename}")

                cap.release()
            else:
                print(f"Warning: Could not parse number from video filename {filename}. Skipping.")

if __name__ == "__main__":
    # 请替换为您实际的视频文件目录
    example_input_directory = sys.argv[1]
    # example_input_directory = "/media/zzg/GJ_disk01/data/Videos/dynerf_test/cook_spinach_4DG/"
    print(f"Attempting to extract frames from: {example_input_directory}")
    extract_first_frame(example_input_directory)
