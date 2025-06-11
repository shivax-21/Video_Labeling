
import cv2
import os
from transformers import pipeline
from PIL import Image # Pillow for image processing

def setup_captioning_pipeline(model_name="Salesforce/blip-image-captioning-base"):
    """
    Sets up the Hugging Face image-to-text pipeline (VLM).
    You can try "Salesforce/blip-image-captioning-large" for potentially better results,
    but it will require more computational resources.
    """
    try:
        # Load the image-to-text pipeline with the chosen VLM model
        # The 'device' parameter can be set to 0 for GPU, or -1 for CPU.
        # If you have a GPU, set device=0 for much faster processing.
        # If you don't have a GPU or encounter CUDA errors, use device=-1.
        caption_pipeline = pipeline("image-to-text", model=model_name, device=0)
        print(f"Successfully loaded VLM model: {model_name}")
        return caption_pipeline
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to CPU if GPU failed. Try setting device=-1 if issues persist.")
        try:
            caption_pipeline = pipeline("image-to-text", model=model_name, device=-1)
            return caption_pipeline
        except Exception as e:
            print(f"Failed to load model even on CPU: {e}")
            return None

def process_video_for_captions(video_path, caption_pipeline, frame_interval=30):
    """
    Extracts frames from a video at a specified interval and generates captions for them.

    Args:
        video_path (str): The path to the input video file.
        caption_pipeline: The Hugging Face image-to-text pipeline.
        frame_interval (int): How often to sample frames (e.g., 30 means every 30th frame).
                              A higher number means fewer frames, faster processing, but less detail.
                              A lower number means more frames, slower processing, but more detail.
                              For typical videos, 30-60 frames (1-2 seconds at 30fps) is a good start.

    Returns:
        list: A list of generated captions for the sampled frames.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}. Check file path and format.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing '{os.path.basename(video_path)}' (FPS: {fps}, Total Frames: {total_frames})")

    frame_captions = []
    current_frame_count = 0

    while True:
        ret, frame = cap.read() # Read a frame from the video

        if not ret: # If no more frames, break the loop
            break

        # Process only selected frames based on frame_interval
        if current_frame_count % frame_interval == 0:
            try:
                # OpenCV reads images in BGR format, VLM models usually expect RGB
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Generate caption for the current frame
                # The pipeline returns a list of dictionaries, we want the 'generated_text'
                caption_result = caption_pipeline(pil_image)
                if caption_result and len(caption_result) > 0:
                    generated_text = caption_result[0]['generated_text']
                    frame_captions.append(f"Frame {current_frame_count}: {generated_text}")
                    print(f"  Generated caption for frame {current_frame_count}")
                else:
                    print(f"  No caption generated for frame {current_frame_count}")
            except Exception as e:
                print(f"  Error captioning frame {current_frame_count}: {e}")
                # Continue processing even if one frame fails

        current_frame_count += 1

    cap.release() # Release the video capture object
    print(f"Finished processing frames for '{os.path.basename(video_path)}'.")
    return frame_captions

def summarize_captions_with_llm(frame_captions):
    """
    Uses an LLM (or a simpler approach for this challenge) to summarize
    a list of individual frame captions into a more coherent video description.

    For this challenge, we'll use a basic summarization method. For more advanced
    summarization, you'd integrate another LLM (e.g., a summarization pipeline
    from Hugging Face, or an external LLM API if you have access and keys).
    """
    if not frame_captions:
        return "No visual content was detected or captioned from the video."

    # Join a selection of captions to give a sense of the video's content.
    # For a real LLM, you'd send these to a summarization model.
    # Example using Hugging Face summarization pipeline (requires 'sentencepiece' and 'accelerate'):
    # from transformers import pipeline
    # summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # full_text = "\n".join(frame_captions)
    # summary_result = summarizer(full_text, max_length=150, min_length=30, do_sample=False)
    # return summary_result[0]['summary_text']

    # Simple concatenation for demonstration:
    # Take a few captions from the beginning, middle, and end to form a general idea.
    selected_captions = []
    if len(frame_captions) > 0:
        selected_captions.append(frame_captions[0]) # First frame
    if len(frame_captions) > 2:
        selected_captions.append(frame_captions[len(frame_captions) // 2]) # Middle frame
    if len(frame_captions) > 1:
        selected_captions.append(frame_captions[-1]) # Last frame

    if len(selected_captions) < 3 and len(frame_captions) > len(selected_captions):
        # If video is short, just include all.
        return "Video content: " + " ".join(frame_captions)

    return "Video content: " + " ".join(selected_captions) + " (A more detailed summary would require an advanced LLM.)"


if __name__ == "__main__":

    video_files = [
        "/content/ApplyEyeMakeup.avi",
        "/content/HandStandPushups.avi",
        "/content/PizzaTossing.avi",
        "/content/SoccerPenalty.avi",
        "/content/CuttingInKitchen.avi",
        "/content/WritingOnBoard.avi",
        "/content/Typing.avi",
        "/content/WalkingWithDog.avi",
        "/content/YoYo.avi",
        "/content/UnevenBars (1).avi"
    ]

    # Ensure the output directory exists
    output_dir = "video_descriptions"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the VLM pipeline once
    caption_pipeline = setup_captioning_pipeline()
    if caption_pipeline is None:
        print("Failed to initialize captioning pipeline. Exiting.")
        exit()

    all_video_results = {}

    for video_file_path in video_files:
        print(f"\n--- Starting processing for: {os.path.basename(video_file_path)} ---")

        # Get frame-level captions
        captions_for_current_video = process_video_for_captions(video_file_path, caption_pipeline, frame_interval=50)

        # Summarize into a single description
        full_video_description = summarize_captions_with_llm(captions_for_current_video)

        print(f"\n--- Generated Description for {os.path.basename(video_file_path)} ---")
        print(full_video_description)

        all_video_results[os.path.basename(video_file_path)] = full_video_description

        # Save the description to a file
        output_filename = os.path.join(output_dir, f"{os.path.basename(video_file_path)}.txt")
        with open(output_filename, "w") as f:
            f.write(full_video_description)
        print(f"Description saved to {output_filename}")

    print("\n\n--- All Video Descriptions Generated ---")
    for video_name, description in all_video_results.items():
        print(f"\nVideo: {video_name}")
        print(f"Description: {description}")


