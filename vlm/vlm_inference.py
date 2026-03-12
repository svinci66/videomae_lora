import os
import cv2
import glob
import base64
import subprocess
import json
from volcenginesdkarkruntime import Ark

class VLMPredictor:
    def __init__(self, api_key=None, base_url=None, model_endpoint_id=None):
        self.api_key = api_key or "0b227c79-ace5-444f-ba30-88f7bf3350e7"
        self.base_url = base_url or 'https://ark.cn-beijing.volces.com/api/v3'
        self.model_endpoint_id = model_endpoint_id or "ep-20260112163413-gnk82"
        self.client = Ark(base_url=self.base_url, api_key=self.api_key)
        
        # 帧标签到抽动大类的映射
        self.label_to_category = {
            'None': None,  # 无抽动
            'face-tic': '面部抽动',
            'head-tic': '头颈抽动',
            'body-tic': '躯体抽动'
        }
    
    def extract_label_from_frame(self, frame_path):
        """从帧文件名中提取标签"""
        base_name = os.path.splitext(os.path.basename(frame_path))[0]
        parts = base_name.rsplit('_', 1)
        if len(parts) >= 2:
            return parts[-1]
        return 'None'
    
    def get_video_category(self, frame_list):
        """根据帧标签列表确定视频的抽动大类
        
        Args:
            frame_list: 帧文件路径列表
            
        Returns:
            tuple: (是否有抽动，抽动大类)
                - 如果全为 None: (False, None)
                - 否则：(True, 抽动大类)
        """
        categories = set()
        for frame_path in frame_list:
            label = self.extract_label_from_frame(frame_path)
            category = self.label_to_category.get(label)
            if category is not None:
                categories.add(category)
        
        if len(categories) == 0:
            return False, None
        elif len(categories) == 1:
            return True, list(categories)[0]
        else:
            # 如果存在多个类别，返回第一个（通常视频应保持单一类别）
            return True, list(categories)[0]
        
    def sanitize_video(self, input_path, output_path):
        """清洗视频 (去除音频 + 规范化)"""
        print(f"  正在清洗视频：{input_path}")
        if os.path.exists(output_path):
            os.remove(output_path)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "+faststart",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"  清洗完成。大小：{os.path.getsize(output_path)/1024:.2f} KB")
        return output_path
    
    def frames_to_video(self, frame_dir, output_video_path, fps=10):
        """将目录下的帧图片合成视频"""
        if os.path.exists(output_video_path):
            print(f"  Video already exists: {output_video_path}")
            return output_video_path, False
        
        frame_list = glob.glob(os.path.join(frame_dir, '*.jpg'))
        frame_list.sort()
        
        if len(frame_list) == 0:
            raise Exception(f"No frames in {frame_dir}")
        
        first_frame = cv2.imread(frame_list[0])
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        for frame_path in frame_list:
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        return output_video_path, True
    
    def call_vlm_api(self, video_path, prompt="请描述这个视频的内容"):
        """调用视觉大模型 API 分析视频"""
        clean_video_path = video_path.replace('.mp4', '_clean.mp4')
        
        try:
            self.sanitize_video(video_path, clean_video_path)
        except subprocess.CalledProcessError:
            print("  FFmpeg 执行失败，使用原始视频")
            clean_video_path = video_path
        
        with open(clean_video_path, "rb") as f:
            video_data = f.read()
            base64_str = base64.b64encode(video_data).decode('utf-8')
        
        print(f"  Base64 长度：{len(base64_str)} 字符")
        print(f"  正在发送请求给模型：{self.model_endpoint_id} ...")
        
        response = self.client.chat.completions.create(
            model=self.model_endpoint_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{base64_str}"
                            }
                        }
                    ]
                }
            ]
        )
        
        result_content = response.choices[0].message.content
        
        if os.path.exists(clean_video_path) and clean_video_path != video_path:
            os.remove(clean_video_path)
        
        return result_content
    
    def process_video(self, video_dir, output_dir, fps=10, prompt="请分析这个视频", keep_video=False):
        """处理单个视频目录"""
        video_filename = os.path.basename(video_dir) + ".mp4"
        video_path = os.path.join(output_dir, video_filename)
        txt_output_path = os.path.join(output_dir, os.path.basename(video_dir) + ".txt")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            video_path, is_newly_created = self.frames_to_video(video_dir, video_path, fps=fps)
            
            if not is_newly_created:
                print(f"  Using existing video: {video_path}")
            else:
                print(f"  Created new video: {video_path}")
            
            result = self.call_vlm_api(video_path, prompt=prompt)
            
            with open(txt_output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"  Result saved: {txt_output_path}")
            
            if is_newly_created and not keep_video:
                os.remove(video_path)
                print(f"  Temp video removed: {video_path}")
            
            return {
                'video_dir': video_dir,
                'video_path': video_path,
                'result_path': txt_output_path,
                'result': result,
                'is_newly_created': is_newly_created
            }
            
        except Exception as e:
            print(f"  Error processing {video_dir}: {str(e)}")
            if os.path.exists(video_path) and not keep_video:
                os.remove(video_path)
            raise
    
    def process_dataset(self, data_dir, txt_file, output_dir, is_test=False, fps=10, prompt="请分析这个视频", keep_video=False, batch_size=5, structured_prompt=None, skip_none=True):
        """处理数据集中的所有视频
        
        Args:
            data_dir: 数据目录
            txt_file: 包含视频列表的 txt 文件
            output_dir: 输出目录
            is_test: 是否为测试集
            fps: 帧率
            prompt: 基础 prompt
            keep_video: 是否保留生成的视频文件
            batch_size: 每批处理的视频数量，处理完后暂停等待确认
            structured_prompt: 结构化 prompt 模板，支持以下占位符：
                              {video_name} - 视频名称
                              {frame_count} - 帧数
                              {duration} - 视频时长
                              {tic_category} - 抽动大类
            skip_none: 是否跳过无抽动 (全为 None 标签) 的视频
        """
        txt_path = os.path.join(data_dir, txt_file)
        
        with open(txt_path) as f:
            contents = f.readlines()
        
        prefix = 'test' if is_test else 'train'
        input_file = [os.path.join(data_dir, prefix, i.rstrip("\n")) for i in contents]
        
        results = []
        skipped_count = 0
        
        for idx, video_dir in enumerate(input_file):
            print(f"Processing [{idx+1}/{len(input_file)}]: {video_dir}")
            
            frame_list = glob.glob(os.path.join(video_dir, '*.jpg'))
            frame_list.sort()
            
            if len(frame_list) == 0:
                print(f"  Skip: No frames in {video_dir}")
                continue
            
            # 根据帧标签确定抽动大类
            has_tic, tic_category = self.get_video_category(frame_list)
            
            if skip_none and not has_tic:
                print(f"  Skip: 无抽动 (全为 None 标签)")
                skipped_count += 1
                continue
            
            print(f"  抽动大类：{tic_category if tic_category else '混合/未知'}")
            
            # 构建最终 prompt
            context = {
                'video_name': os.path.basename(video_dir),
                'frame_count': len(frame_list),
                'duration': len(frame_list) / fps if fps > 0 else 0,
                'tic_category': tic_category if tic_category else '混合类型'
            }
            final_prompt = self._build_prompt(prompt, structured_prompt, context)
            
            try:
                result = self.process_video(
                    video_dir=video_dir,
                    output_dir=output_dir,
                    fps=fps,
                    prompt=final_prompt,
                    keep_video=keep_video
                )
                results.append(result)
                
                # 每 batch_size 个视频暂停一次
                if (idx + 1) % batch_size == 0:
                    print(f"\n{'='*50}")
                    print(f"已处理 {idx+1}/{len(input_file)} 个视频 (跳过 {skipped_count} 个无抽动视频)")
                    print(f"{'='*50}")
                    while True:
                        user_input = input("输入 'c' 继续处理下一批，输入 'q' 退出：").strip().lower()
                        if user_input == 'q':
                            print("用户中止处理。")
                            return results
                        elif user_input == 'c':
                            print(f"\n继续处理...\n{'='*50}\n")
                            break
                        else:
                            print("无效输入，请输入 'c' 或 'q'")
                
            except Exception as e:
                print(f"  Failed: {str(e)}")
                continue
        
        print(f"\n处理完成：共处理 {len(results)} 个视频，跳过 {skipped_count} 个无抽动视频")
        return results
    
    def _build_prompt(self, base_prompt, structured_prompt=None, context=None):
        """构建最终 prompt
        
        Args:
            base_prompt: 基础 prompt
            structured_prompt: 结构化 prompt 模板
            context: 上下文信息字典
        
        Returns:
            最终使用的 prompt
        """
        if structured_prompt:
            try:
                return structured_prompt.format(**(context or {}))
            except KeyError as e:
                print(f"  Warning: 结构化 prompt 缺少占位符 {e}, 使用基础 prompt")
        return base_prompt


def main():
    data_dir = "/data/sj/tic_3_5/data/TIC"
    output_dir = "/data/sj/videomae_hf/videomae_lora/vlm/videos"
    
    predictor = VLMPredictor()
    
    # 结构化 prompt 模板 - 医疗抽动症分类专用
    structured_prompt = """# Role
你是一位专业的医疗视频分析助手，专门用于辅助分类抽动症（Tic Disorders）患者的抽动类型。

# Task
输入的视频片段中**已明确发生了抽动行为**。请仔细分析该视频，理解患者的动作特征，并从指定的分类列表中选出最匹配的抽动类型。

# 已知抽动大类
该视频标注的抽动大类为：**{tic_category}**
请在该大类范围内优先选择最匹配的具体动作。

# Classification Criteria (分类选项)
请严格限制在以下范围内进行选择：
* **面部抽动**：眨眼，斜眼，皱眉，扬眉，张口，伸舌，撅嘴，歪嘴，舔嘴唇，皱鼻子。
* **头颈抽动**：点头，仰头，摇头，转头，斜颈，耸肩。
* **躯体抽动**：动手指，搓手，握拳，动手腕，举臂，伸展手臂，内旋手臂，动脚趾，伸腿，抖腿，踮脚，蹬足，伸膝，屈膝，伸髋，屈髋，挺胸，收腹，扭腰。

# Rules & Constraints
1.  **必须选择**：视频片段中已明确包含抽动，**严禁输出"无抽动"**，你必须给出一个确定的抽动类型。
2.  **精准匹配**：必须且只能从上述 `Classification Criteria` 的具体动作词汇中，选择**唯一一个**最核心、最明显的动作进行输出。
3.  **精简输出**：严格按照给定的输出格式返回结果，禁止添加任何分析过程、解释性语言、寒暄或标点符号的随意变形。

# Output Format
时间 [0,{duration:.2f}]：[抽动类型]

*(示例 1：视频时长 2.5 秒，核心动作是眨眼 -> 返回 `时间 [0,2.50]：眨眼`)*
*(示例 2：视频时长 4.0 秒，核心动作是耸肩 -> 返回 `时间 [0,4.00]：耸肩`)*

# Video Info
* 视频名称：{video_name}
* 帧数：{frame_count}
* 时长：{duration:.2f} 秒

# Input
* 视频输入：<video_file_placeholder>"""
    
    print("Processing test data...")
    test_results = predictor.process_dataset(
        data_dir=data_dir,
        txt_file="test_tic.txt",
        output_dir=output_dir,
        is_test=True,
        fps=10,
        prompt="请分析这个视频",
        keep_video=False,
        batch_size=5,
        structured_prompt=structured_prompt,
        skip_none=True  # 跳过无抽动视频
    )
    
    print(f"Done! Processed {len(test_results)} test videos.")


if __name__ == "__main__":
    main()
