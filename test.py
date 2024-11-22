import streamlit as st
import cv2
import base64
import tempfile
from openai import OpenAI
import json
from datetime import datetime
import os
from typing import List, Dict, Tuple

class VideoProcessor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def calculate_timestamp(self, frame_idx: int, fps: float) -> float:
        """Calculate timestamp in seconds for a given frame"""
        return frame_idx / fps

    def extract_frames(self, video_path: str) -> tuple:
        """Extract frames and video information"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frames = []
        timestamps = []
        frame_count = 0
        
        # Extract one frame per second
        frame_interval = int(fps)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                frames.append(base64_frame)
                timestamps.append(self.calculate_timestamp(frame_count, fps))
            
            frame_count += 1
            
        cap.release()
        
        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'timestamps': timestamps
        }
        
        return frames, video_info

    def generate_descriptions(self, frames: List[str], video_info: Dict, progress_bar=None) -> Dict:
        """Generate descriptions using GPT-4o with correct timestamps"""
        try:
            descriptions = []
            batch_size = 8  # Process 8 frames at a time
            fps = video_info['fps']
            current_time = 0.0
            
            for i in range(0, len(frames), batch_size):
                if progress_bar:
                    progress_bar.progress(min(i / len(frames), 1.0))
                    
                batch_frames = frames[i:i + batch_size]
                
                # Calculate timestamps for this batch
                batch_timestamps = [
                    self.format_timestamp(current_time + j)
                    for j in range(len(batch_frames))
                ]
                
                # Prepare content for API
                content = [
                    {
                        "type": "text",
                        "text": f"""Generate audio descriptions for these sequential video frames for visually impaired viewers.
                        Video time: {self.format_timestamp(current_time)}
                        
                        Requirements:
                        1. Describe essential visual information not conveyed by audio
                        2. Use timestamp format [MM:SS] followed by description
                        3. Be clear, concise, and specific
                        4. Use present tense and active voice
                        5. Focus on important changes and actions
                        6. No interpretation - describe what's visible
                        
                        Format: [MM:SS] Description"""
                    }
                ]
                
                # Add frames to content
                for frame in batch_frames:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": "low"
                        }
                    })
                
                # Get descriptions from GPT-4o
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": content
                    }],
                    max_tokens=500,
                    temperature=0.7
                )
                
                # Parse descriptions with correct timestamps
                result = response.choices[0].message.content.strip().split('\n')
                for j, desc in enumerate(result):
                    if '[' in desc and ']' in desc:
                        description = desc[desc.find(']') + 1:].strip()
                        if description:
                            timestamp = self.format_timestamp(current_time + j)
                            descriptions.append({
                                "timestamp": timestamp,
                                "description": description
                            })
                
                current_time += len(batch_frames)
            
            # Sort and remove duplicate timestamps
            descriptions.sort(key=lambda x: x['timestamp'])
            unique_descriptions = []
            seen_timestamps = set()
            
            for desc in descriptions:
                if desc['timestamp'] not in seen_timestamps:
                    unique_descriptions.append(desc)
                    seen_timestamps.add(desc['timestamp'])
            
            # Create final output
            output_data = {
                "video_path": os.path.basename(video_info.get('path', 'video.mp4')),
                "video_duration": video_info['duration'],
                "processing_date": datetime.now().isoformat(),
                "total_frames_analyzed": len(frames),
                "descriptions": unique_descriptions
            }
            
            return output_data
            
        except Exception as e:
            st.error(f"Error generating descriptions: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Video Audio Description Generator",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Video Audio Description Generator")
    st.write("Generate audio descriptions for visually impaired viewers")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    
    # Video upload
    video_file = st.file_uploader(
        "Upload Video",
        type=['mp4', 'mov', 'avi'],
        help="Upload a video file to generate descriptions"
    )
    
    if video_file and api_key:
        try:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_file.read())
                video_path = tfile.name
            
            # Initialize processor
            processor = VideoProcessor(api_key)
            
            # Process video
            with st.spinner("Extracting video frames..."):
                frames, video_info = processor.extract_frames(video_path)
                video_info['path'] = video_file.name
            
            if frames:
                st.info(f"Extracted {len(frames)} frames for analysis")
                
                # Generate descriptions with progress bar
                progress_bar = st.progress(0)
                with st.spinner("Generating descriptions..."):
                    output_data = processor.generate_descriptions(
                        frames,
                        video_info,
                        progress_bar
                    )
                
                if output_data:
                    st.success("✅ Processing complete!")
                    
                    # Display results
                    st.write("### Generated Descriptions")
                    
                    # Display descriptions in a more readable format
                    for desc in output_data['descriptions']:
                        st.write(f"**[{desc['timestamp']}]** {desc['description']}")
                    
                    # Add download button
                    json_str = json.dumps(output_data, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name="audio_descriptions.json",
                        mime="application/json"
                    )
            
            # Cleanup
            os.unlink(video_path)
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
    
    st.sidebar.write("### About")
    st.sidebar.info("""
        This tool generates timestamped audio descriptions for videos using GPT-4o.
        The descriptions are designed to help visually impaired viewers understand
        the visual content of videos.
    """)

if __name__ == "__main__":
    main()