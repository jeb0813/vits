import argparse
import text
from utils import load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

  parser.add_argument("--reverse", type=bool, default=False,help='reverse processed phoneme')

  args = parser.parse_args()
    

  for filelist in args.filelists:
    print("START:", filelist)
    # 这个函数在加载的同时完成了分割工作
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in range(len(filepaths_and_text)):
      # 这一步取出的是原始文本
      original_text = filepaths_and_text[i][args.text_index]
      # cleaner comes from tacotron
      cleaned_text = text._clean_text(original_text, args.text_cleaners)
      if args.reverse is True:
        cleaned_text=reversed(cleaned_text)
      filepaths_and_text[i][args.text_index] = cleaned_text
    
    # 这里保存的是音素
    if args.reverse is True:
      new_filelist = filelist + "_reverse" + "." + args.out_extension
    else:
      new_filelist = filelist + "." + args.out_extension
    
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
