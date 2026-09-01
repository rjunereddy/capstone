import shutil
import os

files_to_copy = [
    (r"G:\My Drive\Phishguard\Celeb-DF\Celeb-synthesis\id0_id9_0008.mp4", r"c:\Users\malli\capstone\demo_pages\id0_id9_0008.mp4"),
    (r"G:\My Drive\Phishguard\PhishExtracted\1&1 Ionos+2019-07-28-22`34`40\html.txt", r"c:\Users\malli\capstone\demo_pages\ionos-phish.html"),
    (r"G:\My Drive\Phishguard\PhishExtracted\1&1 Ionos+2019-07-28-22`34`40\shot.png", r"c:\Users\malli\capstone\demo_pages\shot.png"),
]

for src, dst in files_to_copy:
    print(f"Copying {src} to {dst}")
    try:
        shutil.copy(src, dst)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

# Process emails
emails = [
    (r"G:\My Drive\Phishguard\email\easy_ham\easy_ham\0002.b3120c4bcbf3101e661161ee7efcb8bf", "email_0002.html"),
    (r"G:\My Drive\Phishguard\email\easy_ham\easy_ham\0003.acfc5ad94bbd27118a0d8685d18c89dd", "email_0003.html"),
    (r"G:\My Drive\Phishguard\email\easy_ham\easy_ham\0004.e8d5727378ddde5c3be181df593f1712", "email_0004.html"),
]

html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Email</title>
    <style>body { font-family: monospace; white-space: pre-wrap; padding: 20px; background: #fff; color: #333; }</style>
</head>
<body>
{content}
</body>
</html>
"""

for src, dst_name in emails:
    dst = os.path.join(r"c:\Users\malli\capstone\demo_pages", dst_name)
    try:
        with open(src, 'r', encoding='latin-1') as f:
            content = f.read()
        
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(html_template.replace('{content}', content.replace('<', '&lt;').replace('>', '&gt;')))
        print(f"Processed email {dst_name}")
    except Exception as e:
        print(f"Failed email {src}: {e}")

