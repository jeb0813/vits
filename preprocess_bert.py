# -*- coding: UTF-8 -*-

__author__ = "Chen Ziang"

# example textfile format
# {wav_path}|{speaker_name}|{language}|{text}

import argparse
import text
from utils import load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=2, type=int)
  parser.add_argument("--language_index", default=1, type=int)
  parser.add_argument("--wav_index", default=0, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

  parser.add_argument("--reverse", type=bool, default=False,help='reverse phoneme')

  args = parser.parse_args()

  for filelist in args.filelists:
    new_filelist = filelist + "." + args.out_extension
    out_file = open(new_filelist, "w", encoding="utf-8")

    print("START:", filelist)
    # 加载函数可以不同改
    # return [[wav_path,language,text],[...],[...]]
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in range(len(filepaths_and_text)):
      # 这一步取出的是原始数据
      wav_path=filepaths_and_text[i][args.wav_index]
      original_text = filepaths_and_text[i][args.text_index]
      language=filepaths_and_text[i][args.language_index]
      # doing the cleaning
      norm_text, phones, tones, word2ph = clean_text(text, language)
      out_file.write("{}|{}|{}|{}|{}|{}\n".format(wav_path,language,norm_text,
                                                    " ".join(phones),
                                                    " ".join([str(i) for i in tones]),
                                                    " ".join([str(i) for i in word2ph]),
                                                    )
                    )
    
    out_file.close()

