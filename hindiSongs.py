from pytube import YouTube
import os


# Function to download YouTube video as audio
def download_audio(url, output_path):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        downloaded_file = audio_stream.download(output_path=output_path)
        base, ext = os.path.splitext(downloaded_file)
        new_file = base + '.mp3'
        os.rename(downloaded_file, new_file)
        print(f"Downloaded and converted: {new_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


# List of YouTube URLs for wedding songs (add more URLs as needed)
song_urls = [
    "https://www.youtube.com/watch?v=_1YGGHFA-p0&pp=ygUXc2FhamFuamkgZ2hhciBhYXllIHNvbmc%3D",
    "https://www.youtube.com/watch?v=gy9D4pz_VxM",
    "https://youtu.be/52deq20h6Q4?si=UnOB6qSn6efMgX2x",
    "https://www.youtube.com/watch?v=dqRnZWrHhEI&t=1179s",
    "https://youtu.be/hdZHD1qDgU0?si=A8Rbuix3yYEY3a7X",
    "https://youtu.be/q0TSHAsNvzA?si=lzHkhXMTCWlphXs8",
    "https://youtu.be/BbTT_QajCNw?si=IPCT1nByWFVp2rtw",
    "https://youtu.be/bk1wAQmy_t4?si=q9aPAnvmaB1fLaqa",
    "https://youtu.be/I35cnkMpNHA?si=Qc4Ddp_B-or33MfK",
    "https://youtube.com/shorts/evemqT_LOdo?si=PvdPvutrZz1_WqB_",
    "https://youtu.be/-bNwqXvMuB8?si=U8-AMeuG_5vwvEG-",
    "https://youtu.be/guHHhEUYAQ4?si=p8fGV7_l4EVnAudj",
    "https://youtu.be/Lh0QJ1lpeDI?si=4Ku-H6JwGRnN2_vA",
    "https://youtu.be/t8QDdOFg1lM?si=V5ujd8n5kElvegvL",
    "https://youtu.be/7w6zBQdNSIc?si=Zta5YuFMVvP11Ei6",
    "https://youtu.be/DLaQ-Hoc1aU?si=O2052IQ94IvVwSPz",
    "https://youtu.be/lA40lMz24LQ?si=964Hr0Rrf1MWRMiY",
    "https://youtu.be/XSz801Qo23U?si=f43mLcAahs7RYOS3",
    "https://youtu.be/-JmCNdL2lq8?si=DOVePhIaPlmvoAtf",
    "https://youtu.be/wOEJ-m9vgIA?si=FU4IeXnTiwRTUUS4"
]

# Output directory
output_directory = "hindi_wedding_albums"

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Download each song
for url in song_urls:
    download_audio(url, output_directory)